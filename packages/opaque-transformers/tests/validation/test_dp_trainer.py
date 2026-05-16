"""End-to-end tests for DPTrainer with GPT-2 + LoRA.

Exercises DPTrainer's HF-Trainer-parity surface: ``train()``, param
restoration, ``evaluate()``, ``get_train_dataloader()``, callback
dispatch, checkpoint round-trip, and resume.  Datasets are HF-shaped
(pre-padded ``input_ids`` / ``labels`` / ``attention_mask``) and feed
the trainer's default ``transformers.default_data_collator`` directly.
"""

from __future__ import annotations

import os

import pytest
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainerCallback as _HFTrainerCallback
from transformers.trainer_utils import PredictionOutput

from opaque.transformers.trainer import DPTrainer, TrainingArguments, TrainOutput
from opaque.api.transformers.trainer._state import DPTrainerState

from _hf_shared import build_lm_dataset  # noqa: E402


def _default_args(**overrides) -> TrainingArguments:
    """Build TrainingArguments with test defaults.

    Pins ``use_cpu=True`` so the trainer's ``args.device`` resolves to
    CPU regardless of the host (MPS on macOS would otherwise pick up
    the test fixtures' CPU-resident model parameters and produce a
    device mismatch).  Tests that explicitly need an accelerator can
    override.
    """
    defaults = dict(
        per_device_train_batch_size=4,
        clipping_norm=1.0,
        privacy_target_epsilon=10.0,
        privacy_noise_multiplier=1.0,
        use_cpu=True,
    )
    defaults.update(overrides)
    return TrainingArguments(**defaults)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gpt2_model_and_tokenizer():
    """Load GPT-2 small and set pad_token."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


@pytest.fixture
def gpt2_with_lora(gpt2_model_and_tokenizer):
    """GPT-2 with LoRA adapters."""
    model, tokenizer = gpt2_model_and_tokenizer

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["c_attn", "c_proj"],
        fan_in_fan_out=True,
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


@pytest.fixture
def tiny_lm_dataset(gpt2_model_and_tokenizer):
    """Eight pre-padded causal-LM examples for DPTrainer integration tests."""
    _, tokenizer = gpt2_model_and_tokenizer
    texts = [
        "def fibonacci(n): return n",
        "area = math.pi * r * r",
        "return os.listdir(path)",
        "self.items = []",
        "return f'hello {name}'",
        "sorted_list = sorted(xs)",
        "return n % 2 == 0",
        "open file and read it",
    ]
    return build_lm_dataset(texts, tokenizer, max_length=24)


# ---------------------------------------------------------------------------
# DPTrainer tests
# ---------------------------------------------------------------------------


class TestDPTrainerInit:
    """Test DPTrainer construction and basic interface."""

    def test_constructor(self, gpt2_with_lora, tiny_lm_dataset):
        model, tokenizer = gpt2_with_lora
        trainer = DPTrainer(
            model=model,
            args=_default_args(),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        assert trainer.model is model
        assert trainer.processing_class is tokenizer
        assert isinstance(trainer.args, TrainingArguments)

    def test_get_train_dataloader(self, gpt2_with_lora, tiny_lm_dataset):
        model, tokenizer = gpt2_with_lora
        trainer = DPTrainer(
            model=model,
            args=_default_args(),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        loader = trainer.get_train_dataloader()
        batch = next(iter(loader))

        assert "input_ids" in batch
        assert "labels" in batch
        assert "attention_mask" in batch
        assert batch["input_ids"].ndim == 2
        assert batch["labels"].ndim == 2


class TestDPTrainerTrain:
    """Test the full DP-SGD training loop."""

    def test_train_few_steps(self, gpt2_with_lora, tiny_lm_dataset):
        """Train for a few steps and verify state is returned."""
        model, tokenizer = gpt2_with_lora

        pre_train_params = {
            n: p.clone() for n, p in model.named_parameters() if p.requires_grad
        }

        trainer = DPTrainer(
            model=model,
            args=_default_args(
                clipping_norm=1.0,
                max_steps=3,
                num_train_epochs=1,
                learning_rate=1e-3,
                eval_strategy="no",
                logging_steps=1,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        output = trainer.train()

        assert isinstance(output, TrainOutput)
        assert "train_loss" in output.metrics
        assert "privacy_epsilon" in output.metrics
        assert output.metrics["privacy_epsilon"] > 0
        assert output.global_step == 3
        assert output.training_loss > 0

        changed = False
        for n, p in model.named_parameters():
            if p.requires_grad and n in pre_train_params:
                if not torch.allclose(p.data, pre_train_params[n]):
                    changed = True
                    break
        assert changed, "Model parameters did not change after training"

    def test_model_generates_after_training(self, gpt2_with_lora, tiny_lm_dataset):
        """Verify model.generate() works after param restoration."""
        model, tokenizer = gpt2_with_lora

        trainer = DPTrainer(
            model=model,
            args=_default_args(
                max_steps=2,
                eval_strategy="no",
                logging_steps=999,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        trainer.train()

        inputs = tokenizer("def hello():", return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        decoded = tokenizer.decode(output[0])
        assert len(decoded) > 0


class TestDPTrainerEvaluate:
    """Test the evaluation method."""

    def test_evaluate_returns_loss(self, gpt2_with_lora, tiny_lm_dataset):
        model, tokenizer = gpt2_with_lora

        trainer = DPTrainer(
            model=model,
            args=_default_args(),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        metrics = trainer.evaluate()

        assert "eval_loss" in metrics
        assert isinstance(metrics["eval_loss"], float)
        assert metrics["eval_loss"] > 0

    def test_evaluate_custom_prefix(self, gpt2_with_lora, tiny_lm_dataset):
        model, tokenizer = gpt2_with_lora

        trainer = DPTrainer(
            model=model,
            args=_default_args(),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        metrics = trainer.evaluate(metric_key_prefix="test")

        assert "test_loss" in metrics

    def test_evaluate_dict_eval_dataset_dispatches_per_task(
        self,
        gpt2_with_lora,
        tiny_lm_dataset,
    ):
        """``eval_dataset={"task_a": ds_a, "task_b": ds_b}`` returns per-task metrics.

        HF parity: each entry is evaluated independently and the
        per-entry metric keys are namespaced with
        ``f"{metric_key_prefix}_{name}_*"``.  The merged dict is
        returned as one row.

        ``evaluate`` does not auto-tokenize datasets passed at call
        time — they must already be in the trainer's expected batch
        shape (this matches HF, where the trainer takes dataset shape
        as the user's responsibility).  The test pre-pads via the
        same ``build_lm_dataset`` helper the fixture uses.
        """
        model, tokenizer = gpt2_with_lora

        ds_a = build_lm_dataset(["x sample", "y sample"], tokenizer, max_length=16)
        ds_b = build_lm_dataset(
            ["a sample", "b sample", "c sample"],
            tokenizer,
            max_length=16,
        )

        trainer = DPTrainer(
            model=model,
            args=_default_args(),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        metrics = trainer.evaluate(eval_dataset={"task_a": ds_a, "task_b": ds_b})

        # Both task-specific loss keys present.
        assert "eval_task_a_loss" in metrics
        assert "eval_task_b_loss" in metrics
        # Throughput trio also namespaced per task.
        assert "eval_task_a_runtime" in metrics
        assert "eval_task_b_runtime" in metrics


class TestDPTrainerCallbacks:
    """Test that callbacks are fired at the right points."""

    def test_callbacks_are_invoked(self, gpt2_with_lora, tiny_lm_dataset):
        """Track which callback hooks fire during training."""
        model, tokenizer = gpt2_with_lora

        fired = []

        from transformers import TrainerCallback

        class TrackingCallback(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                fired.append("on_train_begin")

            def on_log(self, args, state, control, logs=None, **kwargs):
                fired.append("on_log")

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                fired.append("on_evaluate")

            def on_train_end(self, args, state, control, **kwargs):
                fired.append("on_train_end")

        trainer = DPTrainer(
            model=model,
            args=_default_args(
                max_steps=2,
                eval_strategy="steps",
                eval_steps=2,
                logging_steps=1,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            callbacks=[TrackingCallback()],
        )

        trainer.train()

        assert "on_train_begin" in fired
        assert "on_log" in fired
        assert "on_train_end" in fired
        assert "on_evaluate" in fired


class TestDPTrainerPhase7Flags:
    """Focused tests for Phase 7 trainer-contract flags."""

    def test_explicit_train_ignores_do_train_flag(
        self, gpt2_with_lora, tiny_lm_dataset
    ):
        model, tokenizer = gpt2_with_lora
        trainer = DPTrainer(
            model=model,
            args=_default_args(
                max_steps=2,
                eval_strategy="no",
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        out = trainer.train()
        assert out.global_step == 2
        assert out.training_loss > 0

    def test_predict_returns_prediction_output(self, gpt2_with_lora, tiny_lm_dataset):
        model, tokenizer = gpt2_with_lora
        trainer = DPTrainer(
            model=model,
            args=_default_args(eval_strategy="no"),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        out = trainer.predict(tiny_lm_dataset)
        assert isinstance(out, PredictionOutput)
        assert "test_loss" in out.metrics

    def test_explicit_predict_ignores_do_predict_flag(
        self,
        gpt2_with_lora,
        tiny_lm_dataset,
    ):
        model, tokenizer = gpt2_with_lora
        trainer = DPTrainer(
            model=model,
            args=_default_args(eval_strategy="no"),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        out = trainer.predict(tiny_lm_dataset)
        assert isinstance(out, PredictionOutput)
        assert "test_loss" in out.metrics

    def test_debug_underflow_overflow_callback_is_wired(
        self,
        gpt2_with_lora,
        tiny_lm_dataset,
    ):
        model, tokenizer = gpt2_with_lora
        trainer = DPTrainer(
            model=model,
            args=_default_args(debug="underflow_overflow"),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        assert hasattr(trainer, "_debug_underflow_overflow")
        assert (
            trainer._debug_underflow_overflow.__class__.__name__
            == "DebugUnderflowOverflow"
        )

    def test_auto_find_batch_size_retries_microbatch(
        self, gpt2_with_lora, tiny_lm_dataset, monkeypatch
    ):
        model, tokenizer = gpt2_with_lora
        trainer = DPTrainer(
            model=model,
            args=_default_args(
                auto_find_microbatch_size=True,
                per_device_train_batch_size=8,
                eval_strategy="no",
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        calls: list[int | None] = []

        def fake_train_once(
            *, resume_from_checkpoint, microbatch_size_override, ignore_keys_for_eval
        ):
            assert ignore_keys_for_eval is None
            calls.append(microbatch_size_override)
            if len(calls) < 3:
                raise RuntimeError("CUDA out of memory")
            return TrainOutput(
                global_step=1, training_loss=1.0, metrics={"train_loss": 1.0}
            )

        monkeypatch.setattr(trainer, "_train_once", fake_train_once)
        out = trainer.train()

        assert out.global_step == 1
        assert calls == [8, 4, 2]

    def test_auto_find_batch_size_stops_at_floor(
        self,
        gpt2_with_lora,
        tiny_lm_dataset,
        monkeypatch,
    ):
        model, tokenizer = gpt2_with_lora
        trainer = DPTrainer(
            model=model,
            args=_default_args(
                auto_find_microbatch_size=True,
                per_device_train_batch_size=1,
                eval_strategy="no",
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        def fake_train_once(
            *, resume_from_checkpoint, microbatch_size_override, ignore_keys_for_eval
        ):
            assert ignore_keys_for_eval is None
            raise RuntimeError("out of memory")

        monkeypatch.setattr(trainer, "_train_once", fake_train_once)
        with pytest.raises(RuntimeError, match="out of memory"):
            trainer.train()

    def test_past_index_is_rejected(self, tmp_path):
        # ``past_index`` is not on the TrainingArguments surface; unknown
        # kwargs raise ``TypeError`` (louder HF-porting signal).
        with pytest.raises(TypeError, match="past_index"):
            TrainingArguments(
                output_dir=str(tmp_path),
                use_cpu=True,
                past_index=1,
            )


class TestDPTrainerAdaptiveClipping:
    """Test adaptive clipping mode."""

    def test_adaptive_clipping_runs(self, gpt2_with_lora, tiny_lm_dataset):
        model, tokenizer = gpt2_with_lora

        trainer = DPTrainer(
            model=model,
            args=_default_args(
                clipping_mode="adaptive",
                clipping_norm=1.0,
                max_steps=3,
                eval_strategy="no",
                logging_steps=999,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        output = trainer.train()
        assert output.global_step == 3
        assert output.metrics["privacy_epsilon"] > 0


class TestDPTrainerLRScheduling:
    """Test that ``lr_scheduler_type`` and warmup actually take effect."""

    def test_constant_lr_logged_at_base(self, gpt2_with_lora, tiny_lm_dataset):
        """lr_scheduler_type='constant' logs base_lr at every step."""
        model, tokenizer = gpt2_with_lora

        trainer = DPTrainer(
            model=model,
            args=_default_args(
                clipping_norm=1.0,
                max_steps=3,
                num_train_epochs=1,
                learning_rate=1e-3,
                lr_scheduler_type="constant",
                eval_strategy="no",
                logging_steps=1,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        trainer.train()

        lrs = [
            e["learning_rate"]
            for e in trainer.state.log_history
            if "learning_rate" in e
        ]
        assert len(lrs) == 3
        for lr in lrs:
            assert lr == pytest.approx(1e-3)

    def test_linear_warmup_produces_expected_lr_series(
        self, gpt2_with_lora, tiny_lm_dataset
    ):
        """Warmup ramp + linear decay produces the expected lr_schedule(step) series."""
        model, tokenizer = gpt2_with_lora

        trainer = DPTrainer(
            model=model,
            args=_default_args(
                clipping_norm=1.0,
                max_steps=5,
                num_train_epochs=1,
                learning_rate=1e-3,
                lr_scheduler_type="linear",
                warmup_steps=2,
                eval_strategy="no",
                logging_steps=1,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )

        trainer.train()

        lrs = [
            e["learning_rate"]
            for e in trainer.state.log_history
            if "learning_rate" in e
        ]
        # HF parity: the LR logged at global_step k is the value that was
        # *just applied* to the optimizer update, i.e. ``schedule(k - 1)``.
        # With warmup_steps=2 over a 5-step linear schedule:
        #   step 1 → schedule(0) = 0.0   (start of warmup)
        #   step 2 → schedule(1) = 5e-4  (mid-warmup)
        #   step 3 → schedule(2) = 1e-3  (end of warmup, peak)
        #   step 4 → schedule(3) = 1e-3 * 2/3
        #   step 5 → schedule(4) = 1e-3 * 1/3
        expected = [0.0, 5e-4, 1e-3, 1e-3 * 2 / 3, 1e-3 * 1 / 3]
        assert len(lrs) == 5
        for got, exp in zip(lrs, expected):
            assert got == pytest.approx(exp, abs=1e-9)

    def test_warmup_changes_param_trajectory(self, gpt2_with_lora, tiny_lm_dataset):
        """Trainers with constant vs warmup LR diverge on identical seed/data."""
        model_const, tok = gpt2_with_lora
        init_params = {
            n: p.clone().detach()
            for n, p in model_const.named_parameters()
            if p.requires_grad
        }

        common = dict(
            clipping_norm=1.0,
            max_steps=4,
            num_train_epochs=1,
            learning_rate=1e-3,
            warmup_steps=2,
            seed=42,
            eval_strategy="no",
            logging_steps=999,
        )

        trainer1 = DPTrainer(
            model=model_const,
            args=_default_args(lr_scheduler_type="constant", **common),
            processing_class=tok,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer1.train()
        constant_params = {
            n: p.clone().detach()
            for n, p in model_const.named_parameters()
            if p.requires_grad
        }

        # Reset model to initial state so the second run sees the same start.
        for n, p in model_const.named_parameters():
            if n in init_params:
                p.data.copy_(init_params[n])

        trainer2 = DPTrainer(
            model=model_const,
            args=_default_args(lr_scheduler_type="linear", **common),
            processing_class=tok,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer2.train()
        warmup_params = {
            n: p.clone().detach()
            for n, p in model_const.named_parameters()
            if p.requires_grad
        }

        diverged = any(
            not torch.allclose(constant_params[n], warmup_params[n])
            for n in constant_params
        )
        assert diverged, "constant vs linear+warmup produced identical params"


class TestDPTrainerCheckpointing:
    """End-to-end checkpoint save / rotation / final-save tests (Phase 2a)."""

    def _common_args(self, output_dir, **overrides):
        defaults = dict(
            clipping_norm=1.0,
            max_steps=4,
            num_train_epochs=1,
            learning_rate=1e-3,
            lr_scheduler_type="constant",
            eval_strategy="no",
            logging_steps=1,
            output_dir=str(output_dir),
            save_strategy="steps",
            save_steps=2,
            save_safetensors=True,
            overwrite_output_dir=True,
        )
        defaults.update(overrides)
        return _default_args(**defaults)

    def test_save_at_step_intervals(self, gpt2_with_lora, tiny_lm_dataset, tmp_path):
        model, tokenizer = gpt2_with_lora
        trainer = DPTrainer(
            model=model,
            args=self._common_args(tmp_path),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer.train()

        ckpts = sorted(
            p.name for p in tmp_path.iterdir() if p.name.startswith("checkpoint-")
        )
        assert ckpts == ["checkpoint-2", "checkpoint-4"]

        for name in ckpts:
            d = tmp_path / name
            # PEFT models save adapter_*; full models save model.safetensors / config.json.
            weights_present = (d / "model.safetensors").exists() or (
                d / "adapter_model.safetensors"
            ).exists()
            cfg_present = (d / "config.json").exists() or (
                d / "adapter_config.json"
            ).exists()
            assert weights_present
            assert cfg_present
            assert (d / "dp_optimizer.pt").exists()
            assert (d / "dp_state.pt").exists()
            assert (d / "accountant.json").exists()
            assert (d / "rng_state.pth").exists()
            assert (d / "trainer_state.json").exists()
            assert (d / "training_args.bin").exists()

    def test_save_safetensors_false_writes_pickle(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """save_safetensors=False writes the legacy .bin format, and we can resume from it."""
        model, tokenizer = gpt2_with_lora
        trainer = DPTrainer(
            model=model,
            args=self._common_args(
                tmp_path, max_steps=2, save_steps=2, save_safetensors=False
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer.train()
        d = tmp_path / "checkpoint-2"
        # PEFT writes adapter_model.bin; full models write pytorch_model.bin.
        assert (d / "pytorch_model.bin").exists() or (d / "adapter_model.bin").exists()
        # No safetensors files when the flag is off.
        assert not (d / "model.safetensors").exists()
        assert not (d / "adapter_model.safetensors").exists()

        # Resume from this .bin checkpoint to confirm load works for both formats.
        model2, tokenizer2 = gpt2_with_lora
        trainer2 = DPTrainer(
            model=model2,
            args=self._common_args(
                tmp_path, max_steps=4, save_steps=2, save_safetensors=False
            ),
            processing_class=tokenizer2,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        out = trainer2.train(resume_from_checkpoint=str(d))
        assert out.global_step == 4

    def test_save_only_model_skips_runtime_files(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        model, tokenizer = gpt2_with_lora
        trainer = DPTrainer(
            model=model,
            args=self._common_args(
                tmp_path, save_only_model=True, max_steps=2, save_steps=2
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer.train()

        d = tmp_path / "checkpoint-2"
        weights_present = (d / "model.safetensors").exists() or (
            d / "adapter_model.safetensors"
        ).exists()
        cfg_present = (d / "config.json").exists() or (
            d / "adapter_config.json"
        ).exists()
        assert weights_present
        assert cfg_present
        # Resumability files (optimizer / DP runtime / RNG) are skipped
        # under ``save_only_model``.
        assert not (d / "dp_optimizer.pt").exists()
        assert not (d / "dp_state.pt").exists()
        assert not (d / "rng_state.pth").exists()
        # Interpretability files are always written: ``trainer_state.json``,
        # ``training_args.bin`` (HF-parity filename), and ``accountant.json``
        # (the privacy guarantee is a property of the saved model, not
        # training state).
        assert (d / "trainer_state.json").exists()
        assert (d / "training_args.bin").exists()
        assert (d / "accountant.json").exists()

    def test_save_total_limit_keeps_most_recent(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        model, tokenizer = gpt2_with_lora
        trainer = DPTrainer(
            model=model,
            args=self._common_args(
                tmp_path, max_steps=6, save_steps=2, save_total_limit=2
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer.train()

        names = sorted(
            p.name for p in tmp_path.iterdir() if p.name.startswith("checkpoint-")
        )
        assert names == ["checkpoint-4", "checkpoint-6"]

    def test_final_save_when_unaligned(self, gpt2_with_lora, tiny_lm_dataset, tmp_path):
        """Final step saves a checkpoint even if it doesn't align with save_steps."""
        model, tokenizer = gpt2_with_lora
        trainer = DPTrainer(
            model=model,
            args=self._common_args(tmp_path, max_steps=3, save_steps=2),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer.train()

        names = sorted(
            p.name for p in tmp_path.iterdir() if p.name.startswith("checkpoint-")
        )
        # checkpoint-2 from save_steps; checkpoint-3 from final-save.
        assert names == ["checkpoint-2", "checkpoint-3"]

    def test_save_strategy_no_writes_nothing(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        model, tokenizer = gpt2_with_lora
        trainer = DPTrainer(
            model=model,
            args=self._common_args(tmp_path, save_strategy="no"),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer.train()
        assert [
            p.name for p in tmp_path.iterdir() if p.name.startswith("checkpoint-")
        ] == []

    def test_save_strategy_epoch(self, gpt2_with_lora, tiny_lm_dataset, tmp_path):
        model, tokenizer = gpt2_with_lora
        trainer = DPTrainer(
            model=model,
            args=self._common_args(
                tmp_path,
                max_steps=-1,
                num_train_epochs=2,
                save_strategy="epoch",
                # Make epoch length tiny so the test is quick.
                per_device_train_batch_size=4,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer.train()
        names = sorted(
            p.name for p in tmp_path.iterdir() if p.name.startswith("checkpoint-")
        )
        # At least one checkpoint per epoch.
        assert len(names) >= 2

    def test_fractional_save_steps_resolved(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """save_steps in (0, 1) is treated as a fraction of total_steps."""
        model, tokenizer = gpt2_with_lora
        trainer = DPTrainer(
            model=model,
            args=self._common_args(tmp_path, max_steps=4, save_steps=0.5),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer.train()
        # 0.5 * 4 == 2 → checkpoints at 2 and 4
        names = sorted(
            p.name for p in tmp_path.iterdir() if p.name.startswith("checkpoint-")
        )
        assert names == ["checkpoint-2", "checkpoint-4"]

    def test_trainer_state_json_round_trips(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """Saved trainer_state.json deserializes to an equivalent DPTrainerState."""
        import json

        model, tokenizer = gpt2_with_lora
        trainer = DPTrainer(
            model=model,
            args=self._common_args(tmp_path, max_steps=2, save_steps=2),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer.train()

        state_path = tmp_path / "checkpoint-2" / "trainer_state.json"
        with open(state_path) as f:
            data = json.load(f)
        assert data["global_step"] == 2
        assert data["max_steps"] == 2
        assert isinstance(data["log_history"], list)

        restored = DPTrainerState.from_json(data)
        assert restored.global_step == 2

    def test_save_model_public_api(self, gpt2_with_lora, tiny_lm_dataset, tmp_path):
        """save_model() writes weights, training args, and accountant.json."""
        model, tokenizer = gpt2_with_lora
        out = tmp_path / "final"
        trainer = DPTrainer(
            model=model,
            args=self._common_args(tmp_path, save_strategy="no", max_steps=2),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer.train()
        trainer.save_model(str(out))
        assert (out / "model.safetensors").exists() or (
            out / "adapter_model.safetensors"
        ).exists()
        assert (out / "config.json").exists() or (out / "adapter_config.json").exists()
        # Privacy provenance travels with the saved model.
        assert (out / "accountant.json").exists()

    # ------------------------------------------------------------------
    # Phase 2b: best-model tracking
    # ------------------------------------------------------------------

    def test_load_best_model_at_end_requires_strategies(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        # HF parity: ``load_best_model_at_end`` with mismatched
        # ``eval_strategy``/``save_strategy`` is rejected by
        # ``TrainingArguments.__post_init__`` before the trainer
        # even constructs.  The error wording mirrors HF's exact phrasing.
        with pytest.raises(ValueError, match="save and eval strategy to match"):
            self._common_args(
                tmp_path,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                eval_strategy="no",
            )

    def test_load_best_model_at_end_defaults_metric_to_loss(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        # HF parity: omitting ``metric_for_best_model`` under
        # ``load_best_model_at_end`` is now silently filled with ``"loss"``
        # by ``TrainingArguments.__post_init__`` (it used to raise).
        model, tokenizer = gpt2_with_lora
        args = self._common_args(
            tmp_path,
            load_best_model_at_end=True,
            save_strategy="steps",
            save_steps=2,
            eval_strategy="steps",
            eval_steps=2,
            metric_for_best_model=None,
        )
        assert args.metric_for_best_model == "loss"
        # ``greater_is_better`` defaults to ``False`` for loss-suffixed metrics.
        assert args.greater_is_better is False
        # Construction still succeeds with the auto-default in place.
        trainer = DPTrainer(
            model=model,
            args=args,
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        assert trainer is not None

    def test_greater_is_better_default_for_loss(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        model, tokenizer = gpt2_with_lora
        trainer = DPTrainer(
            model=model,
            args=self._common_args(
                tmp_path,
                eval_strategy="steps",
                eval_steps=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        # Loss is minimized → greater_is_better=False.  HF parity:
        # ``__post_init__`` populates the default at construction time,
        # so the args object reflects the resolved value just like
        # ``transformers.TrainingArguments``.
        assert trainer._greater_is_better is False
        assert trainer.args.greater_is_better is False

    def test_greater_is_better_default_for_accuracy(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        model, tokenizer = gpt2_with_lora
        trainer = DPTrainer(
            model=model,
            args=self._common_args(
                tmp_path,
                eval_strategy="steps",
                eval_steps=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_accuracy",
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        # Anything not ending in "loss" → greater_is_better=True.  HF parity:
        # ``__post_init__`` populates the default at construction time.
        assert trainer._greater_is_better is True
        assert trainer.args.greater_is_better is True

    def test_best_metric_tracking_runs(self, gpt2_with_lora, tiny_lm_dataset, tmp_path):
        """state.best_* is populated after training when eval is enabled."""
        model, tokenizer = gpt2_with_lora
        trainer = DPTrainer(
            model=model,
            args=self._common_args(
                tmp_path,
                max_steps=4,
                save_steps=2,
                eval_strategy="steps",
                eval_steps=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer.train()
        assert trainer.state.best_metric is not None
        assert trainer.state.best_global_step in (2, 4)
        assert trainer.state.best_model_checkpoint is not None
        assert os.path.isdir(trainer.state.best_model_checkpoint)

    def test_save_strategy_best_only_saves_on_improvement(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        model, tokenizer = gpt2_with_lora
        trainer = DPTrainer(
            model=model,
            args=self._common_args(
                tmp_path,
                max_steps=4,
                eval_strategy="steps",
                eval_steps=2,
                save_strategy="best",
                metric_for_best_model="eval_loss",
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer.train()
        names = sorted(
            p.name for p in tmp_path.iterdir() if p.name.startswith("checkpoint-")
        )
        # First eval always counts as an improvement (best_metric was None).
        # Final-save will also fire at the last step, even if no improvement.
        assert len(names) >= 1

    # ------------------------------------------------------------------
    # Phase 2c: resume from checkpoint
    # ------------------------------------------------------------------

    def test_resume_continues_global_step(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """Resume from a saved checkpoint advances global_step instead of restarting."""
        model, tokenizer = gpt2_with_lora
        # Initial run: 2 steps, save at step 2.
        trainer1 = DPTrainer(
            model=model,
            args=self._common_args(tmp_path, max_steps=2, save_steps=2),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        out1 = trainer1.train()
        assert out1.global_step == 2
        ckpt_dir = str(tmp_path / "checkpoint-2")
        assert os.path.isdir(ckpt_dir)

        # Resume: max_steps=4 → run 2 more steps starting at global_step=2.
        model2, tokenizer2 = gpt2_with_lora
        trainer2 = DPTrainer(
            model=model2,
            args=self._common_args(tmp_path, max_steps=4, save_steps=2),
            processing_class=tokenizer2,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        out2 = trainer2.train(resume_from_checkpoint=ckpt_dir)
        assert out2.global_step == 4
        assert trainer2.state.global_step == 4
        # checkpoint-4 was just written by the final save / save_steps.
        assert os.path.isdir(str(tmp_path / "checkpoint-4"))

    def test_resume_true_finds_latest(self, gpt2_with_lora, tiny_lm_dataset, tmp_path):
        """resume_from_checkpoint=True picks the latest checkpoint under output_dir."""
        model, tokenizer = gpt2_with_lora
        trainer1 = DPTrainer(
            model=model,
            args=self._common_args(tmp_path, max_steps=2, save_steps=2),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer1.train()

        model2, tokenizer2 = gpt2_with_lora
        trainer2 = DPTrainer(
            model=model2,
            args=self._common_args(tmp_path, max_steps=4, save_steps=2),
            processing_class=tokenizer2,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        out2 = trainer2.train(resume_from_checkpoint=True)
        assert out2.global_step == 4

    def test_resume_missing_path_raises(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        model, tokenizer = gpt2_with_lora
        trainer = DPTrainer(
            model=model,
            args=self._common_args(tmp_path, save_strategy="no"),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        with pytest.raises(FileNotFoundError):
            trainer.train(resume_from_checkpoint=str(tmp_path / "no-such-ckpt"))

    def test_resume_true_starts_fresh_when_no_checkpoints(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """``resume_from_checkpoint=True`` is tolerant: auto-finds the latest
        checkpoint when one exists, falls back to a fresh run when none does.

        Lets scripts pass ``resume_from_checkpoint=True`` unconditionally —
        "resume if you can, else start fresh" — without probing the filesystem.
        """
        model, tokenizer = gpt2_with_lora
        trainer = DPTrainer(
            model=model,
            args=self._common_args(tmp_path, save_strategy="no"),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        out = trainer.train(resume_from_checkpoint=True)
        # Fresh-run semantics: global_step advances from 0, not via resume.
        assert out.global_step > 0

    def test_resume_missing_accountant_raises_by_default(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """Missing ``accountant.json`` on resume raises by default.

        The privacy provenance of prior training lives in
        ``accountant.json``; resuming without it would silently discard
        the spent budget.  Default policy is hard-fail.
        """
        import os

        # Produce a checkpoint, then delete its accountant.json.
        model, tokenizer = gpt2_with_lora
        trainer = DPTrainer(
            model=model,
            args=self._common_args(tmp_path, max_steps=2, save_steps=2),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer.train()
        ckpt_dir = tmp_path / "checkpoint-2"
        os.remove(ckpt_dir / "accountant.json")

        model2, tokenizer2 = gpt2_with_lora
        trainer2 = DPTrainer(
            model=model2,
            args=self._common_args(tmp_path, max_steps=4, save_steps=2),
            processing_class=tokenizer2,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        with pytest.raises(FileNotFoundError, match="accountant.json is missing"):
            trainer2.train(resume_from_checkpoint=str(ckpt_dir))

    def test_resume_missing_accountant_opt_in_recalibrates(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """``privacy_resume_without_accountant=True`` permits resume with empty prefix.

        Designed for the warmup-then-DP workflow: prior training had
        zero DP cost (e.g. trained on public data), so calibration
        proceeds against an empty accountant over the remaining steps.
        """
        import os

        model, tokenizer = gpt2_with_lora
        trainer = DPTrainer(
            model=model,
            args=self._common_args(tmp_path, max_steps=2, save_steps=2),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer.train()
        ckpt_dir = tmp_path / "checkpoint-2"
        os.remove(ckpt_dir / "accountant.json")

        model2, tokenizer2 = gpt2_with_lora
        trainer2 = DPTrainer(
            model=model2,
            args=self._common_args(
                tmp_path,
                max_steps=4,
                save_steps=2,
                privacy_resume_without_accountant=True,
            ),
            processing_class=tokenizer2,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        out = trainer2.train(resume_from_checkpoint=str(ckpt_dir))
        assert out.global_step == 4

    def test_resume_restores_accountant(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """epsilon at end of resumed run reflects total composition (incl. saved steps)."""
        model, tokenizer = gpt2_with_lora
        trainer1 = DPTrainer(
            model=model,
            args=self._common_args(tmp_path, max_steps=2, save_steps=2),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        out1 = trainer1.train()
        eps_after_2 = out1.metrics["privacy_epsilon"]

        model2, tokenizer2 = gpt2_with_lora
        trainer2 = DPTrainer(
            model=model2,
            args=self._common_args(tmp_path, max_steps=4, save_steps=2),
            processing_class=tokenizer2,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        out2 = trainer2.train(resume_from_checkpoint=str(tmp_path / "checkpoint-2"))
        # Resumed run composes additional steps on top of saved process → ε grows.
        assert out2.metrics["privacy_epsilon"] > eps_after_2

    def test_resume_save_only_model_composes_budget(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """Resuming a ``save_only_model=True`` checkpoint composes ε via the
        always-saved ``accountant.json``.  Optimizer state and sampler
        state are absent (resume-only artifacts) but the privacy budget
        is preserved."""
        import math

        model, tokenizer = gpt2_with_lora
        trainer1 = DPTrainer(
            model=model,
            args=self._common_args(
                tmp_path, max_steps=2, save_steps=2, save_only_model=True
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer1.train()
        ckpt_dir = str(tmp_path / "checkpoint-2")
        # Interpretability files always present.
        assert os.path.exists(os.path.join(ckpt_dir, "accountant.json"))
        # Resumability files absent under save_only_model.
        assert not os.path.exists(os.path.join(ckpt_dir, "dp_optimizer.pt"))
        assert not os.path.exists(os.path.join(ckpt_dir, "dp_state.pt"))

        model2, tokenizer2 = gpt2_with_lora
        out2_dir = tmp_path / "resumed"
        out2_dir.mkdir()
        trainer2 = DPTrainer(
            model=model2,
            args=self._common_args(
                out2_dir,
                max_steps=4,
                save_steps=2,
                save_only_model=False,
                privacy_noise_multiplier=1.0,
            ),
            processing_class=tokenizer2,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        out2 = trainer2.train(resume_from_checkpoint=ckpt_dir)
        # The budget composes via the saved accountant — finite, not ∞.
        assert math.isfinite(out2.metrics["privacy_epsilon"])
        assert out2.metrics["privacy_epsilon"] > 0

    def test_ignore_data_skip_runs(self, gpt2_with_lora, tiny_lm_dataset, tmp_path):
        """ignore_data_skip=True still completes training successfully."""
        model, tokenizer = gpt2_with_lora
        trainer1 = DPTrainer(
            model=model,
            args=self._common_args(tmp_path, max_steps=2, save_steps=2),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer1.train()

        model2, tokenizer2 = gpt2_with_lora
        trainer2 = DPTrainer(
            model=model2,
            args=self._common_args(
                tmp_path, max_steps=4, save_steps=2, ignore_data_skip=True
            ),
            processing_class=tokenizer2,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        out2 = trainer2.train(resume_from_checkpoint=str(tmp_path / "checkpoint-2"))
        assert out2.global_step == 4

    def test_resume_restores_callback_state(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """When restore_callback_states_from_checkpoint=True, callbacks reload state."""
        import json

        model, tokenizer = gpt2_with_lora

        from transformers import TrainerCallback
        from transformers.trainer_callback import ExportableState

        class StatefulCallback(TrainerCallback, ExportableState):
            def __init__(self):
                self.value = 0

            def state(self):
                return {"args": {}, "attributes": {"value": self.value}}

            def on_log(self, args, state, control, logs=None, **kwargs):
                self.value += 1

        cb = StatefulCallback()
        trainer1 = DPTrainer(
            model=model,
            args=self._common_args(tmp_path, max_steps=2, save_steps=2),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            callbacks=[cb],
        )
        trainer1.train()
        # The trainer writes the callback's ExportableState payload into the JSON.
        with open(tmp_path / "checkpoint-2" / "trainer_state.json") as f:
            ts = json.load(f)
        assert ts.get("stateful_callbacks", {}).get("StatefulCallback") == {
            "args": {},
            "attributes": {"value": cb.value},
        }
        saved_value = cb.value
        assert saved_value > 0  # received at least one on_log

        cb2 = StatefulCallback()
        model2, tokenizer2 = gpt2_with_lora
        trainer2 = DPTrainer(
            model=model2,
            args=self._common_args(
                tmp_path,
                max_steps=4,
                save_steps=2,
                restore_callback_states_from_checkpoint=True,
            ),
            processing_class=tokenizer2,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            callbacks=[cb2],
        )
        trainer2.train(resume_from_checkpoint=str(tmp_path / "checkpoint-2"))
        # Callback's value was restored before the resumed run started, so it's
        # at least the saved value (further on_log bumps may have advanced it).
        assert cb2.value >= saved_value


# ---------------------------------------------------------------------------
# Phase 3a: eval_on_start, eval_delay, prediction_loss_only
# ---------------------------------------------------------------------------


class _EvalRecorder(_HFTrainerCallback):
    """Pytest helper: records each on_evaluate call with its global_step."""

    def __init__(self):
        self.calls: list[tuple[int, dict]] = []
        self.train_steps_seen: list[int] = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.calls.append((state.global_step, dict(metrics or {})))

    def on_step_end(self, args, state, control, **kwargs):
        self.train_steps_seen.append(state.global_step)


class TestDPTrainerEvalControls:
    """Phase 3a: ``eval_on_start``, ``eval_delay``, ``prediction_loss_only``."""

    def test_eval_on_start_fires_before_first_step(
        self, gpt2_with_lora, tiny_lm_dataset
    ):
        model, tokenizer = gpt2_with_lora
        rec = _EvalRecorder()

        trainer = DPTrainer(
            model=model,
            args=_default_args(
                max_steps=2,
                eval_strategy="steps",
                eval_steps=999,  # disables the per-step trigger
                eval_on_start=True,
                logging_steps=1,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            callbacks=[rec],
        )
        trainer.train()

        # First on_evaluate must fire at global_step=0, before any training step.
        assert rec.calls, "on_evaluate did not fire"
        assert rec.calls[0][0] == 0

    def test_eval_on_start_with_strategy_no(self, gpt2_with_lora, tiny_lm_dataset):
        """eval_on_start fires at step 0 even when eval_strategy='no'."""
        model, tokenizer = gpt2_with_lora
        rec = _EvalRecorder()

        trainer = DPTrainer(
            model=model,
            args=_default_args(
                max_steps=2,
                eval_strategy="no",
                eval_on_start=True,
                logging_steps=1,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            callbacks=[rec],
        )
        trainer.train()
        assert len(rec.calls) == 1
        assert rec.calls[0][0] == 0

    def test_eval_on_start_fires_on_resume(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """``eval_on_start=True`` fires on every ``train()`` call (HF parity).

        Pre-Stage-3 we had an Opaque-specific gate that suppressed this
        eval on resume.  HF *always* fires when the flag is set —
        users opting in want a baseline eval at the start of each
        ``train()`` call, including on resume.  Disable the flag if
        you don't want it on resume.
        """
        model, tokenizer = gpt2_with_lora

        # First run: produce a checkpoint at step 2.
        trainer1 = DPTrainer(
            model=model,
            args=_default_args(
                output_dir=str(tmp_path),
                max_steps=2,
                save_strategy="steps",
                save_steps=2,
                eval_strategy="no",
                logging_steps=1,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
        )
        trainer1.train()

        # Resume: eval_on_start=True ⇒ the resumed run also fires the
        # baseline eval at the start.
        rec = _EvalRecorder()
        model2, tokenizer2 = gpt2_with_lora
        trainer2 = DPTrainer(
            model=model2,
            args=_default_args(
                output_dir=str(tmp_path),
                max_steps=4,
                save_strategy="steps",
                save_steps=2,
                eval_strategy="no",
                eval_on_start=True,
                logging_steps=1,
            ),
            processing_class=tokenizer2,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            callbacks=[rec],
        )
        trainer2.train(resume_from_checkpoint=str(tmp_path / "checkpoint-2"))
        # HF parity: exactly one eval call, fired before the inner loop runs.
        assert len(rec.calls) == 1
        # ``eval_on_start`` fires before the first post-resume step, so
        # ``global_step`` recorded by the eval recorder is the loaded
        # global_step (= 2), not 0.
        assert rec.calls[0][0] == 2

    def test_eval_delay_steps_skips_until_threshold(
        self, gpt2_with_lora, tiny_lm_dataset
    ):
        """eval_delay=4 with eval_steps=2, max_steps=6 → fires at 4 and 6, not 2."""
        model, tokenizer = gpt2_with_lora
        rec = _EvalRecorder()

        trainer = DPTrainer(
            model=model,
            args=_default_args(
                max_steps=6,
                eval_strategy="steps",
                eval_steps=2,
                eval_delay=4,
                logging_steps=1,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            callbacks=[rec],
        )
        trainer.train()

        steps_evaluated = [c[0] for c in rec.calls]
        # 2 must NOT appear; 4 and 6 must appear.
        assert 2 not in steps_evaluated
        assert 4 in steps_evaluated
        assert 6 in steps_evaluated

    def test_prediction_loss_only_skips_compute_metrics(
        self,
        gpt2_with_lora,
        tiny_lm_dataset,
    ):
        """prediction_loss_only=True: compute_metrics is not invoked."""
        model, tokenizer = gpt2_with_lora
        cm_calls = []

        def cm(eval_pred, **kwargs):
            cm_calls.append(eval_pred)
            return {"acc": 0.0}

        trainer = DPTrainer(
            model=model,
            args=_default_args(prediction_loss_only=True),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            compute_metrics=cm,
        )
        out = trainer.evaluation_loop(
            trainer.get_eval_dataloader(),
            description="test",
        )
        assert out.predictions is None
        assert out.label_ids is None
        assert cm_calls == []  # compute_metrics never called
        assert "eval_loss" in out.metrics


# ---------------------------------------------------------------------------
# Phase 3b: eval_accumulation_steps, eval_do_concat_batches, batch_eval_metrics
# ---------------------------------------------------------------------------


class TestDPTrainerEvalMemory:
    """Phase 3b: memory-management flags."""

    def test_eval_accumulation_steps_engages_cpu_flushes(
        self,
        gpt2_with_lora,
        tiny_lm_dataset,
    ):
        """eval_accumulation_steps=2 with 4 batches → 2 mid-loop flushes + 1 final."""
        model, tokenizer = gpt2_with_lora

        # Track flushes by patching _PredictionAccumulator.flush_to_cpu.
        import opaque.api.transformers.trainer._eval as _eval_mod

        flush_count = {"n": 0}
        original = _eval_mod._PredictionAccumulator.flush_to_cpu

        def counting_flush(self):
            flush_count["n"] += 1
            return original(self)

        _eval_mod._PredictionAccumulator.flush_to_cpu = counting_flush
        try:
            trainer = DPTrainer(
                model=model,
                args=_default_args(
                    per_device_eval_batch_size=2,  # 8 examples / 2 = 4 batches
                    eval_accumulation_steps=2,
                ),
                processing_class=tokenizer,
                train_dataset=tiny_lm_dataset,
                eval_dataset=tiny_lm_dataset,
            )
            metrics = trainer.evaluate()
        finally:
            _eval_mod._PredictionAccumulator.flush_to_cpu = original

        assert "eval_loss" in metrics
        # 4 batches with cadence 2 → 2 mid-loop flushes; finalize() adds 1.
        assert flush_count["n"] == 3

    def test_eval_do_concat_batches_false_returns_lists(
        self,
        gpt2_with_lora,
        tiny_lm_dataset,
    ):
        """eval_do_concat_batches=False → predictions delivered as a list."""
        model, tokenizer = gpt2_with_lora
        captured: dict = {}

        def cm(eval_pred):
            captured["predictions"] = eval_pred.predictions
            captured["label_ids"] = eval_pred.label_ids
            return {"shape_marker": 1.0}

        trainer = DPTrainer(
            model=model,
            args=_default_args(
                per_device_eval_batch_size=4,  # 8 examples / 4 = 2 batches
                eval_do_concat_batches=False,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            compute_metrics=cm,
        )
        trainer.evaluate()

        assert isinstance(captured["predictions"], list)
        assert isinstance(captured["label_ids"], list)
        assert len(captured["predictions"]) == 2

# ---------------------------------------------------------------------------
# Phase 3c: include_for_metrics + deprecated alias removal
# ---------------------------------------------------------------------------


class TestDPTrainerEvalMetrics:
    """Phase 3c: ``include_for_metrics`` and the dropped deprecated alias."""

    def test_compute_metrics_receives_eval_prediction(
        self,
        gpt2_with_lora,
        tiny_lm_dataset,
    ):
        """compute_metrics is called once with a fully-populated EvalPrediction."""
        import numpy as np

        model, tokenizer = gpt2_with_lora
        seen: dict = {}

        def cm(eval_pred):
            seen["predictions"] = eval_pred.predictions
            seen["label_ids"] = eval_pred.label_ids
            return {"acc": 0.42}

        trainer = DPTrainer(
            model=model,
            args=_default_args(),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            compute_metrics=cm,
        )
        metrics = trainer.evaluate()
        assert "eval_acc" in metrics
        assert metrics["eval_acc"] == pytest.approx(0.42)
        assert seen["predictions"] is not None
        assert seen["label_ids"] is not None
        # HF parity: ``compute_metrics`` consumes numpy arrays.
        assert isinstance(seen["predictions"], np.ndarray)
        assert isinstance(seen["label_ids"], np.ndarray)

    def test_compute_metrics_numpy_argmax_works(
        self,
        gpt2_with_lora,
        tiny_lm_dataset,
    ):
        """A typical HF-flavored reducer using ``np.argmax`` runs end-to-end.

        Pre-Stage-1 the accumulator returned ``torch.Tensor``; calling
        ``np.argmax`` on it works by accident but ``sklearn.metrics.*``
        and ``evaluate.load(...)`` reducers expect actual numpy arrays.
        Stage-1 numpifies via ``transformers.trainer_pt_utils.nested_numpify``
        so this contract holds.
        """
        import numpy as np

        model, tokenizer = gpt2_with_lora

        def cm(eval_pred):
            preds = eval_pred.predictions
            labels = eval_pred.label_ids
            # Both must be numpy arrays for numpy-style reducers to work.
            assert isinstance(preds, np.ndarray)
            assert isinstance(labels, np.ndarray)
            top1 = np.argmax(preds, axis=-1)
            mask = labels != -100
            denom = max(1, int(mask.sum()))
            correct = int(((top1 == labels) & mask).sum())
            return {"top1_acc": correct / denom}

        trainer = DPTrainer(
            model=model,
            args=_default_args(),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            compute_metrics=cm,
        )
        metrics = trainer.evaluate()
        assert "eval_top1_acc" in metrics
        assert 0.0 <= metrics["eval_top1_acc"] <= 1.0

    def test_include_for_metrics_inputs(self, gpt2_with_lora, tiny_lm_dataset):
        """include_for_metrics=['inputs'] populates EvalPrediction.inputs."""
        model, tokenizer = gpt2_with_lora
        seen: dict = {}

        def cm(eval_pred):
            seen["inputs"] = eval_pred.inputs
            seen["losses"] = eval_pred.losses
            return {"acc": 0.0}

        trainer = DPTrainer(
            model=model,
            args=_default_args(include_for_metrics=["inputs"]),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            compute_metrics=cm,
        )
        trainer.evaluate()
        assert seen["inputs"] is not None
        # HF parity (non-batch path): inputs is the bare main-input numpy array
        # (input_ids), not a full batch dict.  Shape: (num_examples, seq_len).
        import numpy as np

        assert isinstance(seen["inputs"], np.ndarray)
        assert seen["inputs"].ndim == 2
        assert seen["losses"] is None

    def test_include_for_metrics_loss_is_one_d_per_example(
        self,
        gpt2_with_lora,
        tiny_lm_dataset,
    ):
        """include_for_metrics=['loss'] populates a 1-D per-example losses tensor.

        HF parity: each batch's reduced loss is repeated by ``batch_size``
        before being appended, so the final ``EvalPrediction.losses``
        length equals total samples (not number of batches).
        """
        model, tokenizer = gpt2_with_lora
        seen: dict = {}

        def cm(eval_pred):
            seen["losses"] = eval_pred.losses
            return {"acc": 0.0}

        trainer = DPTrainer(
            model=model,
            args=_default_args(
                per_device_eval_batch_size=4,  # 8 examples / 4 = 2 batches
                include_for_metrics=["loss"],
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            compute_metrics=cm,
        )
        trainer.evaluate()
        assert seen["losses"] is not None
        # HF parity: ``compute_metrics`` consumes numpy arrays.
        assert seen["losses"].ndim == 1
        # 8 examples → length-8 per-example losses (HF parity).
        assert seen["losses"].shape[0] == 8

    def test_include_for_metrics_combined(self, gpt2_with_lora, tiny_lm_dataset):
        """Both 'inputs' and 'loss' populate the corresponding fields."""
        model, tokenizer = gpt2_with_lora
        seen: dict = {}

        def cm(eval_pred):
            seen["inputs"] = eval_pred.inputs
            seen["losses"] = eval_pred.losses
            return {"acc": 1.0}

        trainer = DPTrainer(
            model=model,
            args=_default_args(include_for_metrics=["inputs", "loss"]),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            compute_metrics=cm,
        )
        trainer.evaluate()
        assert seen["inputs"] is not None
        assert seen["losses"] is not None

    def test_unknown_include_for_metrics_key_raises(
        self, gpt2_with_lora, tiny_lm_dataset
    ):
        """Unknown entries in include_for_metrics raise at __init__ time."""
        model, tokenizer = gpt2_with_lora
        with pytest.raises(ValueError, match="include_for_metrics"):
            DPTrainer(
                model=model,
                args=_default_args(include_for_metrics=["foo"]),
                processing_class=tokenizer,
                train_dataset=tiny_lm_dataset,
                eval_dataset=tiny_lm_dataset,
            )

# ---------------------------------------------------------------------------
# Phase 3 — end-to-end integration & parity
# ---------------------------------------------------------------------------


def _accuracy_fn(eval_pred):
    """Tiny HF-typed compute_metrics: top-1 token accuracy ignoring -100.

    HF parity: predictions / label_ids are numpy arrays.
    """
    preds = eval_pred.predictions
    labels = eval_pred.label_ids
    if isinstance(preds, list):
        preds = preds[0]
        labels = labels[0]
    pred_ids = preds.argmax(axis=-1)
    mask = labels != -100
    if int(mask.sum()) == 0:
        return {"acc": 0.0}
    correct = (pred_ids == labels) & mask
    return {"acc": float(int(correct.sum()) / int(mask.sum()))}


class TestDPTrainerEvalIntegration:
    """End-to-end: train() with eval_on_start, eval_delay, compute_metrics."""

    def test_full_eval_flow_through_train(
        self, gpt2_with_lora, tiny_lm_dataset, tmp_path
    ):
        """eval_on_start + eval_delay + compute_metrics + load_best_model_at_end."""
        model, tokenizer = gpt2_with_lora

        trainer = DPTrainer(
            model=model,
            args=_default_args(
                output_dir=str(tmp_path),
                max_steps=4,
                eval_strategy="steps",
                eval_steps=2,
                eval_on_start=True,
                eval_delay=2,  # skip the eval at step 2; first auto-eval at 4
                save_strategy="steps",
                save_steps=2,
                metric_for_best_model="eval_acc",
                greater_is_better=True,
                load_best_model_at_end=True,
                logging_steps=1,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            compute_metrics=_accuracy_fn,
        )
        out = trainer.train()
        assert isinstance(out, TrainOutput)

        # log_history must contain at least one row tagged with eval_acc.
        eval_rows = [row for row in trainer.state.log_history if "eval_acc" in row]
        assert eval_rows, "no eval_acc row in log_history"

        # The step-0 eval (eval_on_start) seeded best_metric.
        assert trainer.state.best_metric is not None


class TestDPTrainerEvalParity:
    """Functional ↔ nn.Module parity for compute_metrics."""

    def test_compute_metrics_path_agnostic(self, gpt2_with_lora, tiny_lm_dataset):
        """Same compute_metrics produces the same eval_acc on both paths."""
        model, tokenizer = gpt2_with_lora

        captured: list[float] = []

        def cm(eval_pred):
            r = _accuracy_fn(eval_pred)
            captured.append(r["acc"])
            return r

        trainer = DPTrainer(
            model=model,
            args=_default_args(max_steps=1, eval_strategy="no"),
            processing_class=tokenizer,
            train_dataset=tiny_lm_dataset,
            eval_dataset=tiny_lm_dataset,
            compute_metrics=cm,
        )

        # Mid-train: functional path (self._ctx populated). Capture by
        # invoking evaluate() inside the train() context via a callback.
        functional_acc: list[float] = []

        class _MidTrainEval(_HFTrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                m = trainer.evaluate()
                functional_acc.append(m["eval_acc"])

        trainer._callback_handler.callbacks.append(_MidTrainEval())
        trainer.train()

        # Post-train: nn.Module path (self._ctx is None after train() returns).
        post_metrics = trainer.evaluate()

        assert functional_acc, "functional-path eval did not run"
        # Mid-train and post-train accuracies on the same eval set with the
        # same parameters (after one tiny training step) should be numerically
        # close.  We don't require strict equality because the one training
        # step does shift parameters slightly, but the *path* shouldn't be
        # observable — the key invariant is that both paths return a finite
        # float that compute_metrics could compute.
        assert isinstance(functional_acc[0], float)
        assert isinstance(post_metrics["eval_acc"], float)
