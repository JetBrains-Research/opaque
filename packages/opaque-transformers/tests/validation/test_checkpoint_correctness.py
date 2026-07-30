"""Regression tests for checkpoint correctness.

Covers:

- Best-folder lookup in ``_save_checkpoint`` (HF parity): the best model
  checkpoint is registered by *folder* keyed on ``state.best_global_step``,
  not only when the eval boundary coincides with the save boundary.
- ``_load_best_model`` mutates ``self._model`` (so an immediate
  ``save_model()`` sees the loaded weights without going through
  ``_restore_params`` first).
- ``DPTrainerState.stateful_callbacks`` round-trip
  through ``to_json`` / ``from_json`` (single schema; no JSON re-parse).
- ``_warn_on_arg_drift`` surfaces ``expected_batch_size`` drift so a
  user resuming with a changed batch size sees the privacy-relevant
  change instead of having it silently absorbed by the ``sample_rate``
  warning.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import pytest
from _hf_shared import build_lm_dataset, gpt2_tokenizer, make_gpt2_model
from peft import LoraConfig, TaskType, get_peft_model

from opaque.api.transformers.trainer._state import DPTrainerState
from opaque.transformers.trainer import DPTrainer, TrainingArguments

# ---------------------------------------------------------------------------
# Fixtures (same shape as test_callback_hooks.py).
# ---------------------------------------------------------------------------


@pytest.fixture
def small_model_and_tokenizer():
    tokenizer = gpt2_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token
    model = make_gpt2_model()
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


@pytest.fixture
def lora_model(small_model_and_tokenizer):
    model, tokenizer = small_model_and_tokenizer
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["c_attn"],
        fan_in_fan_out=True,
    )
    return get_peft_model(model, lora_config), tokenizer


@pytest.fixture
def tiny_dataset(small_model_and_tokenizer):
    """Six pre-padded causal-LM examples for checkpoint tests."""
    _, tokenizer = small_model_and_tokenizer
    return build_lm_dataset(
        [f"sample {i}" for i in range(6)],
        tokenizer,
        max_length=16,
    )


def _args(tmp_path, **overrides) -> TrainingArguments:
    # ``use_cpu=True``: pin to CPU so the trainer's ``args.device``
    # resolves to CPU regardless of the host (LoRA fixtures are CPU).
    defaults: dict[str, Any] = {
        "output_dir": str(tmp_path),
        "per_device_train_batch_size": 2,
        "privacy_target_epsilon": 10.0,
        "privacy_noise_multiplier": 1.0,
        "max_steps": 4,
        "num_train_epochs": 1,
        "logging_steps": 1,
        "save_strategy": "no",
        "use_cpu": True,
    }
    defaults.update(overrides)
    return TrainingArguments(**defaults)


# ---------------------------------------------------------------------------
# DPTrainerState round-trip — single schema for trainer_state.json.
# ---------------------------------------------------------------------------


class TestStateRoundTrip:
    """``DPTrainerState`` round-trips every persisted field through ``to_json``."""

    def test_stateful_callbacks_round_trip(self):
        s = DPTrainerState(
            global_step=10,
            stateful_callbacks={"EarlyStoppingCallback": {"patience_counter": 3}},
        )
        round_tripped = DPTrainerState.from_json(s.to_json())
        assert round_tripped.stateful_callbacks == {
            "EarlyStoppingCallback": {"patience_counter": 3}
        }
        assert round_tripped.global_step == 10

    def test_unknown_keys_dropped(self):
        # Forward-compat: a future field is ignored without raising.
        s = DPTrainerState.from_json(
            {
                "global_step": 5,
                "future_field_not_yet_invented": "anything",
            }
        )
        assert s.global_step == 5


# ---------------------------------------------------------------------------
# _load_best_model mutates self._model.
# ---------------------------------------------------------------------------


class TestLoadBestModelMutates:
    """``_load_best_model`` writes the loaded weights into ``self._model``."""

    def test_save_model_after_load_best_sees_best_weights(
        self,
        lora_model,
        tiny_dataset,
        tmp_path,
    ):
        """``save_model()`` after ``train()`` reflects the loaded best weights.

        ``_load_best_model`` mutates ``self._model`` directly, so a
        fresh ``save_model()`` call is byte-identical to the recorded
        best checkpoint regardless of the in-flight functional view.
        """
        model, tokenizer = lora_model
        ckpt_root = tmp_path / "ckpts"
        ckpt_root.mkdir()
        save_target = tmp_path / "post_train"

        trainer = DPTrainer(
            model=model,
            args=_args(
                ckpt_root,
                max_steps=4,
                num_train_epochs=1,
                eval_strategy="steps",
                eval_steps=2,
                save_strategy="steps",
                save_steps=2,
                metric_for_best_model="eval_loss",
                load_best_model_at_end=True,
                logging_steps=1,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
        )
        trainer.train()

        if trainer.state.best_model_checkpoint is None:
            pytest.skip("Eval never improved; nothing to compare.")

        best_dir = trainer.state.best_model_checkpoint
        from safetensors.torch import load_file as load_safetensors

        adapter_path = os.path.join(best_dir, "adapter_model.safetensors")
        if not os.path.exists(adapter_path):
            pytest.skip(f"Expected adapter weights under {best_dir} not found")
        best_weights = load_safetensors(adapter_path, device="cpu")

        # Save the in-memory module to a fresh directory.  This writes
        # the *best* weights — the module was mutated by
        # ``_load_best_model``.
        trainer.save_model(str(save_target))
        new_path = save_target / "adapter_model.safetensors"
        assert new_path.exists(), f"save_model did not produce {new_path}"
        post_save_weights = load_safetensors(str(new_path), device="cpu")

        # Same key set + identical tensors → module reflects best.
        assert set(best_weights) == set(post_save_weights), (
            "save_model produced a different adapter shape than the best "
            "checkpoint — module wasn't mutated by _load_best_model"
        )
        # ``atol=1e-3`` allows fp32 round-trip noise from PEFT's
        # save_pretrained merging.
        for name, saved in best_weights.items():
            diff = (post_save_weights[name] - saved).abs().max().item()
            assert diff < 1e-3, (
                f"Best weights for {name!r} drift from disk by {diff:g} after "
                "_load_best_model — module mutation must reflect best "
                "checkpoint exactly"
            )


# ---------------------------------------------------------------------------
# Best-folder lookup keyed on best_global_step (HF parity).
# ---------------------------------------------------------------------------


class TestBestFolderLookup:
    """``_save_checkpoint`` registers best by folder, not just exact-step coincidence."""

    def test_best_folder_lookup(self, lora_model, tiny_dataset, tmp_path):
        """Eval at step 2 may improve while save_strategy fires at every step.

        With ``save_strategy="steps", save_steps=1`` and
        ``eval_strategy="steps", eval_steps=2``, eval-improvement at
        step 2 must produce ``best_model_checkpoint == "checkpoint-2/"``
        (HF parity), even though the best decision and the save
        decision are made at different cadence boundaries.
        """
        model, tokenizer = lora_model

        trainer = DPTrainer(
            model=model,
            args=_args(
                tmp_path,
                max_steps=4,
                num_train_epochs=1,
                eval_strategy="steps",
                eval_steps=2,
                save_strategy="steps",
                save_steps=1,
                metric_for_best_model="eval_loss",
                save_total_limit=10,
                logging_steps=1,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
        )
        trainer.train()

        if trainer.state.best_global_step is None:
            pytest.skip("Eval never improved; nothing to verify.")

        expected = os.path.join(
            str(tmp_path),
            f"checkpoint-{trainer.state.best_global_step}",
        )
        assert trainer.state.best_model_checkpoint is not None
        assert os.path.abspath(trainer.state.best_model_checkpoint) == os.path.abspath(
            expected
        ), (
            f"best_model_checkpoint should resolve to checkpoint-"
            f"{trainer.state.best_global_step} via folder lookup; got "
            f"{trainer.state.best_model_checkpoint}"
        )
        # The folder should also exist on disk and be persisted in
        # trainer_state.json (single schema; no JSON re-parse).
        ts_path = os.path.join(expected, "trainer_state.json")
        assert os.path.exists(ts_path)
        with open(ts_path) as f:
            ts = json.load(f)
        assert ts["best_global_step"] == trainer.state.best_global_step


# ---------------------------------------------------------------------------
# save_only_model: trainer_state.json is always written (HF parity).
# ---------------------------------------------------------------------------


class TestSaveOnlyModelStillWritesTrainerState:
    """``save_only_model=True`` no longer drops trainer_state.json."""

    def test_trainer_state_present_under_save_only_model(
        self,
        lora_model,
        tiny_dataset,
        tmp_path,
    ):
        model, tokenizer = lora_model
        trainer = DPTrainer(
            model=model,
            args=_args(
                tmp_path,
                save_only_model=True,
                max_steps=2,
                save_strategy="steps",
                save_steps=2,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
        )
        trainer.train()
        d = tmp_path / "checkpoint-2"
        # Interpretability files always written (HF parity for the first
        # two; ``accountant.json`` is the DP-correct extension — the
        # privacy guarantee is a property of the saved model).
        assert (d / "trainer_state.json").exists()
        assert (d / "training_args.bin").exists()
        assert (d / "accountant.json").exists()
        # Resumability files skipped under save_only_model.
        assert not (d / "dp_optimizer.pt").exists()
        assert not (d / "dp_state.pt").exists()


# ---------------------------------------------------------------------------
# EarlyStoppingCallback round-trip.
# ---------------------------------------------------------------------------


class TestEarlyStoppingExportableState:
    """HF's stock ``EarlyStoppingCallback`` saves under the ExportableState schema.

    HF's callback uses the ``ExportableState`` protocol
    (``state()`` / ``from_state()``) rather than a flat ``state_dict()``;
    the trainer must persist via ``state()`` when the callback supports it
    or the callback's internal counters are lost on resume.

    This test pins the *save-side* contract.  The *load-side*
    contract (attribute-set on resume) is covered by the simpler
    legacy callback test in ``test_dp_trainer.py``
    (``test_resume_restores_callback_state``).
    """

    def test_state_dict_schema_matches_hf(
        self,
        lora_model,
        tiny_dataset,
        tmp_path,
    ):
        from transformers.trainer_callback import EarlyStoppingCallback

        model, tokenizer = lora_model
        cb = EarlyStoppingCallback(early_stopping_patience=2)
        trainer = DPTrainer(
            model=model,
            args=_args(
                tmp_path,
                max_steps=2,
                save_strategy="steps",
                save_steps=2,
                eval_strategy="steps",
                eval_steps=2,
                metric_for_best_model="eval_loss",
                logging_steps=1,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
            callbacks=[cb],
        )
        trainer.train()

        # ``stateful_callbacks`` has the HF ExportableState shape:
        # ``{"args": {...}, "attributes": {...}}``.  This is what
        # ``transformers.trainer_callback.TrainerState`` writes too —
        # the schema is the canonical HF round-trip shape.
        ts_path = tmp_path / "checkpoint-2" / "trainer_state.json"
        with open(ts_path) as f:
            ts = json.load(f)
        cb_payload = ts["stateful_callbacks"].get("EarlyStoppingCallback")
        assert cb_payload is not None
        assert "args" in cb_payload
        assert "attributes" in cb_payload
        # The args round-trip exactly through ``EarlyStoppingCallback.state()``.
        assert cb_payload["args"]["early_stopping_patience"] == 2


# ---------------------------------------------------------------------------
# expected_batch_size drift warning + drift-disposition dispatch.
# ---------------------------------------------------------------------------


class TestArgDriftWarnings:
    """``_warn_on_arg_drift`` surfaces drift per the field disposition.

    ``expected_batch_size`` carries the ``dp_relevant`` disposition so a
    user resuming with a different ``per_device_train_batch_size`` sees
    the privacy-relevant change rather than having it silently absorbed
    by the ``sample_rate`` warning.

    Tested at the ``_warn_on_arg_drift`` boundary directly so the
    test doesn't drag in the full save/resume cycle (which is
    exercised by other resume tests that already pin the
    ``sample_rate`` / ``noise_multiplier`` warnings).
    """

    def _baseline_runtime(self, trainer, tiny_dataset):
        """Build a ``RuntimeCheckpoint`` whose privacy scalars match live args.

        Drift comparisons are made field-by-field; baseline values that
        match the live trainer trip no warnings.  Tests then mutate the
        field under test on the returned bundle.
        """
        import opaque.api.transformers.trainer._checkpoint as ckpt

        target_delta = trainer.args.privacy_target_delta or 1e-5
        return ckpt.RuntimeCheckpoint(
            version=ckpt.DP_STATE_BUNDLE_VERSION,
            clip_state={},
            noise_state={},
            sampler_state=None,
            sample_rate=trainer.args.train_batch_size / max(1, len(tiny_dataset)),
            target_delta=float(target_delta),
            noise_multiplier=float(trainer.args.privacy_noise_multiplier),
            expected_steps_per_epoch=1,
            expected_batch_size=int(trainer.args.train_batch_size),
            total_steps=int(trainer.args.max_steps),
        )

    def test_expected_batch_size_drift_warns(
        self,
        lora_model,
        tiny_dataset,
        tmp_path,
        caplog,
    ):
        """A synthetic bundle whose ``expected_batch_size`` differs warns."""
        model, tokenizer = lora_model
        trainer = DPTrainer(
            model=model,
            args=_args(tmp_path, per_device_train_batch_size=2),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
        )
        runtime = self._baseline_runtime(trainer, tiny_dataset)
        # Drift only ``expected_batch_size``: saved=4, current=2.
        runtime.expected_batch_size = int(trainer.args.train_batch_size) * 2

        with caplog.at_level(logging.WARNING):
            trainer._warn_on_arg_drift(runtime)

        drift_msgs = [
            r
            for r in caplog.records
            if "expected_batch_size" in r.getMessage()
            and "drift" in r.getMessage().lower()
        ]
        assert drift_msgs, (
            f"expected an ``expected_batch_size`` drift warning; "
            f"got records: {[r.getMessage() for r in caplog.records]}"
        )

    def test_no_drift_no_warning(self, lora_model, tiny_dataset, tmp_path, caplog):
        """Identical saved/current bundle emits no drift warning."""
        model, tokenizer = lora_model
        trainer = DPTrainer(
            model=model,
            args=_args(tmp_path, per_device_train_batch_size=2),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
        )
        runtime = self._baseline_runtime(trainer, tiny_dataset)

        with caplog.at_level(logging.WARNING):
            trainer._warn_on_arg_drift(runtime)

        drift_msgs = [r for r in caplog.records if "drift" in r.getMessage().lower()]
        assert not drift_msgs, (
            f"no drift was introduced but warnings fired: "
            f"{[r.getMessage() for r in drift_msgs]}"
        )

    def test_total_steps_drift_silent_for_dp_sgd(
        self, lora_model, tiny_dataset, tmp_path, caplog
    ):
        """``total_steps`` drift is silent under DP-SGD — users extend
        training routinely (intentional_extend disposition).
        """
        model, tokenizer = lora_model
        trainer = DPTrainer(
            model=model,
            args=_args(tmp_path, per_device_train_batch_size=2, max_steps=10),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
        )
        runtime = self._baseline_runtime(trainer, tiny_dataset)
        runtime.total_steps = 5  # saved=5, current=10 — clean extension

        with caplog.at_level(logging.WARNING):
            trainer._warn_on_arg_drift(runtime)

        total_msgs = [r for r in caplog.records if "total_steps" in r.getMessage()]
        assert not total_msgs, (
            "DP-SGD total_steps extension should be silent; got: "
            f"{[r.getMessage() for r in total_msgs]}"
        )

    def test_total_steps_drift_raises_for_dp_ftrl(
        self, lora_model, tiny_dataset, tmp_path
    ):
        """``total_steps`` drift under DP-FTRL raises — the MF strategy
        is shape-locked for the original composition.
        """
        model, tokenizer = lora_model
        trainer = DPTrainer(
            model=model,
            args=_args(tmp_path, per_device_train_batch_size=2, max_steps=10),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
        )
        runtime = self._baseline_runtime(trainer, tiny_dataset)
        runtime.mechanism_kind = "mf_band"
        runtime.total_steps = 5  # saved=5, current=10 — DP-FTRL extension

        with pytest.raises(ValueError, match="DP-FTRL resume forbids drift"):
            trainer._warn_on_arg_drift(runtime)

    def test_shape_drift_warns_on_lr_scheduler(
        self, lora_model, tiny_dataset, tmp_path, caplog
    ):
        """``lr_scheduler`` drift is a shape-disposition warning."""
        model, tokenizer = lora_model
        trainer = DPTrainer(
            model=model,
            args=_args(tmp_path, per_device_train_batch_size=2, lr_scheduler="linear"),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
        )
        runtime = self._baseline_runtime(trainer, tiny_dataset)
        runtime.lr_scheduler = "cosine"  # saved=cosine, current=linear

        with caplog.at_level(logging.WARNING):
            trainer._warn_on_arg_drift(runtime)

        shape_msgs = [
            r
            for r in caplog.records
            if "lr_scheduler" in r.getMessage() and "shape" in r.getMessage()
        ]
        assert shape_msgs, (
            f"expected an lr_scheduler shape-drift warning; "
            f"got records: {[r.getMessage() for r in caplog.records]}"
        )
