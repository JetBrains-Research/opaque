"""Optimizer name validation, HF-name rejection, and the torchopt set.

DPTrainer's optimizer chain is functional (``torchopt``); HF's
``OptimizerNames`` enum binds to ``torch.optim`` / 8-bit / fused-CUDA
implementations whose state shapes and update math do not match
torchopt's.  Silently substituting ``torchopt.adamw`` for
``torch.optim.AdamW`` (HF's ``adamw_torch``) would produce wrong
numerics, so DPTrainer rejects every HF-specific name with a
per-name redirection rather than aliasing it.

This test module covers:

- HF-specific names (``adamw_torch``, ``adamw_hf``, ``adafactor``,
  ``lion_32bit``, ``adamw_bnb_8bit``, …) raise ``ValueError`` with a
  message that names the optimizer and points at a torchopt-backed
  alternative.
- All names DPTrainer's factory wires (``adam``, ``adamw``,
  ``adamw-bc``, ``sgd``, ``rmsprop``, ``adagrad``, ``adadelta``,
  ``adamax``, ``radam``) construct successfully and run one step.
- ``optim='sgd', weight_decay=…`` actually decays the params (the
  SGD branch previously dropped ``weight_decay`` silently).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.training_args import OptimizerNames

from opaque.transformers.trainer import DPTrainer, DPTrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _hf_shared import build_lm_dataset  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gpt2_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def _build_lora_model(base_model, seed: int = 42):
    """LoRA-wrap a base GPT-2 with a fixed adapter shape (PEFT-seeded)."""
    from transformers import set_seed

    set_seed(seed)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["c_attn"],
        fan_in_fan_out=True,
    )
    return get_peft_model(base_model, lora_config)


def _args(tmp_path, **overrides) -> DPTrainingArguments:
    """Optimizer-test args (CPU-pinned, σ=0 for bit-comparable runs)."""
    defaults = dict(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        max_steps=3,
        num_train_epochs=1,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        dp_target_epsilon=10.0,
        dp_noise_multiplier=0.0,
        dp_clipping_norm=1.0,
        learning_rate=1e-2,
        seed=42,
        use_cpu=True,
    )
    defaults.update(overrides)
    return DPTrainingArguments(**defaults)


# ---------------------------------------------------------------------------
# HF-name rejection
# ---------------------------------------------------------------------------


class TestOptimRejectsHFNames:
    """HF ``OptimizerNames`` values raise — DPTrainer is torchopt-only.

    The redirection is meant to save users the time of debugging why
    their HF script silently misbehaves.  The error message must
    (a) name the unsupported optimizer, (b) reference torchopt or the
    redirection, and (c) point at the supported list.
    """

    @pytest.mark.parametrize(
        "name",
        [
            # HF default — ``torch.optim.AdamW``, *not* equivalent to
            # ``torchopt.adamw``.  This is the most important reject:
            # silently substituting one for the other would change
            # the numerics of every HF training script ported as-is.
            "adamw_torch",
            "adamw_torch_fused",
            "adamw_torch_xla",
            "adamw_torch_npu_fused",
            # Pre-PyTorch-2 HF default.
            "adamw_hf",
            # APEX / quantized / specialised AdamW variants.
            "adamw_apex_fused",
            "adamw_anyprecision",
            "adamw_bnb_8bit",
            "adamw_8bit",
            "adamw_torch_4bit",
            "adamw_torch_8bit",
            # Adafactor — factored second moments don't compose under
            # vmap.
            "adafactor",
            # AdEMAMix / Lion / paged variants.
            "ademamix",
            "ademamix_8bit",
            "lion_32bit",
            "lion_8bit",
            "paged_adamw_32bit",
            "paged_adamw_8bit",
            "paged_lion_32bit",
            "paged_lion_8bit",
            # bitsandbytes RMSprop variants — DPTrainer ships its own
            # ``rmsprop`` (torchopt-backed).
            "rmsprop_bnb",
            "rmsprop_bnb_8bit",
            "rmsprop_bnb_32bit",
            # GaLore / LOMO / schedule-free / APOLLO / StableAdamW —
            # all out of scope.
            "galore_adamw",
            "galore_adafactor",
            "lomo",
            "adalomo",
            "grokadamw",
            "schedule_free_adamw",
            "schedule_free_sgd",
            "apollo_adamw",
            "stable_adamw",
        ],
    )
    def test_hf_specific_optim_raises(self, tmp_path, name):
        with pytest.raises(ValueError) as exc_info:
            _args(tmp_path, optim=name)
        message = str(exc_info.value)
        assert name in message, (
            f"error must name the rejected optimizer; got: {message}"
        )
        assert "DPTrainer" in message
        # The redirection mentions either the supported list or a
        # specific torchopt-backed alternative.
        assert "Supported optimizers" in message or "torchopt" in message

    def test_optimizernames_enum_input_also_raises(self, tmp_path):
        """``optim=OptimizerNames.ADAMW_TORCH`` must raise too."""
        # Until Step 3 was rewound this enum input was silently
        # folded onto ``adamw``.  After the rewind the enum's
        # ``.value`` (``'adamw_torch'``) lands in the rejection
        # table and raises.
        with pytest.raises(ValueError, match="adamw_torch"):
            _args(tmp_path, optim=OptimizerNames.ADAMW_TORCH)

    def test_unknown_optim_still_raises(self, tmp_path):
        """Names absent from both the rejection table and the supported
        list fall through to the bare validation step."""
        with pytest.raises(ValueError, match="expected one of"):
            _args(tmp_path, optim="nonsense_optim")


# ---------------------------------------------------------------------------
# Supported torchopt-backed optimizers
# ---------------------------------------------------------------------------


# All torchopt-backed names DPTrainer ships.  Each must (a) construct
# the args without raising and (b) advance one training step
# end-to-end.  ``adamw-bc`` is excluded from the smoke run because
# it requires a ``clip_state`` payload that isn't trivially available
# at args-construction time; it is exercised by the dedicated DP-SGD
# tests under ``packages/opaque-dpsgd/tests/``.
SUPPORTED_OPTIMIZERS = (
    "adam",
    "adamw",
    "adamw-bc",
    "sgd",
    "rmsprop",
    "adagrad",
    "adadelta",
    "adamax",
    "radam",
)

SMOKE_OPTIMIZERS = tuple(n for n in SUPPORTED_OPTIMIZERS if n != "adamw-bc")


class TestSupportedOptimizersConstruct:
    """All torchopt-backed names accepted by ``DPTrainingArguments``."""

    @pytest.mark.parametrize("name", SUPPORTED_OPTIMIZERS)
    def test_native_name_constructs(self, tmp_path, name):
        args = _args(tmp_path, optim=name)
        assert args.optim == name

    @pytest.mark.parametrize("name", SMOKE_OPTIMIZERS)
    def test_optimizer_runs_one_step(
        self, tmp_path, gpt2_model_and_tokenizer, name
    ):
        """End-to-end smoke: optimizer constructs *and* trains.

        Verifies that ``create_optimizer``'s dispatch chain wires
        every advertised name to a torchopt factory that successfully
        completes ``opt.init(trainable_params)`` and one update.
        """
        base, tokenizer = gpt2_model_and_tokenizer
        model = _build_lora_model(base, seed=42)
        dataset = build_lm_dataset(
            ["hello world test", "another sample"],
            tokenizer,
            max_length=8,
        )
        args = _args(tmp_path, optim=name, max_steps=1)
        trainer = DPTrainer(
            model=model,
            args=args,
            processing_class=tokenizer,
            train_dataset=dataset,
            eval_dataset=dataset,
        )
        trainer.train()
        assert trainer.state.global_step == 1


# ---------------------------------------------------------------------------
# SGD weight-decay wiring
# ---------------------------------------------------------------------------


class TestSgdWeightDecay:
    """``optim='sgd', weight_decay=…`` actually decays the params.

    Pre-Stage-3 the SGD branch of :meth:`DPTrainer.create_optimizer`
    silently dropped ``weight_decay`` (HF parity bug).  After Stage 3
    we forward ``weight_decay=args.weight_decay`` to ``torchopt.sgd``
    and a non-zero decay produces measurably different param norms
    from a zero-decay run on the same seed.
    """

    def test_weight_decay_changes_param_norm(
        self, tmp_path, gpt2_model_and_tokenizer
    ):
        """Two SGD runs at σ=0 with different weight_decay diverge."""
        base, tokenizer = gpt2_model_and_tokenizer
        dataset = build_lm_dataset(
            ["hello world", "another sample", "third sample", "fourth one"],
            tokenizer,
            max_length=8,
        )

        def _run(weight_decay: float) -> torch.Tensor:
            # Re-seed PEFT init so both runs start from the same LoRA
            # adapter weights — any divergence below must come from
            # the optimizer step, not from differing init.
            # Deep-copy so get_peft_model doesn't mutate the shared
            # fixture model on the second call (which triggers a PEFT
            # "modifying for a second time" UserWarning).
            import copy
            model = _build_lora_model(copy.deepcopy(base), seed=42)
            args = _args(
                tmp_path,
                optim="sgd",
                weight_decay=weight_decay,
                # SGD with momentum=0 and a substantive learning rate
                # so weight decay has visible effect over 3 steps.
                learning_rate=1e-1,
                max_steps=3,
            )
            trainer = DPTrainer(
                model=model,
                args=args,
                processing_class=tokenizer,
                train_dataset=dataset,
                eval_dataset=dataset,
            )
            trainer.train()
            # ``trainer._ctx`` is cleared at the end of ``train()``;
            # the canonical post-training param snapshot lives on the
            # underlying module via ``_restore_params``-style writeback,
            # so we read ``model.named_parameters`` (HF parity).
            return torch.cat(
                [
                    p.detach().cpu().flatten()
                    for _, p in model.named_parameters()
                    if p.requires_grad
                ]
            )

        without_decay = _run(weight_decay=0.0)
        with_decay = _run(weight_decay=1e-1)

        # ``allclose`` would pass on identical tensors; we want to see
        # them *diverge*.  L2 distance is the cleanest summary.
        diff = (without_decay - with_decay).norm().item()
        assert diff > 1e-4, (
            f"SGD weight_decay had no effect: "
            f"||params(wd=0) - params(wd=0.1)||={diff:g}.  "
            "Stage 3 should have wired weight_decay into torchopt.sgd."
        )

    def test_zero_weight_decay_is_noop(
        self, tmp_path, gpt2_model_and_tokenizer
    ):
        """``weight_decay=0`` is the implicit default; same result twice."""
        base, tokenizer = gpt2_model_and_tokenizer
        dataset = build_lm_dataset(
            ["hello world", "another sample"], tokenizer, max_length=8,
        )

        def _run() -> torch.Tensor:
            import copy
            model = _build_lora_model(copy.deepcopy(base), seed=42)
            args = _args(
                tmp_path,
                optim="sgd",
                weight_decay=0.0,
                learning_rate=1e-1,
                max_steps=2,
            )
            trainer = DPTrainer(
                model=model,
                args=args,
                processing_class=tokenizer,
                train_dataset=dataset,
                eval_dataset=dataset,
            )
            trainer.train()
            return torch.cat(
                [
                    p.detach().cpu().flatten()
                    for _, p in model.named_parameters()
                    if p.requires_grad
                ]
            )

        a = _run()
        b = _run()
        # σ=0 + same seed + ``weight_decay=0`` must reproduce.
        assert torch.allclose(a, b, atol=1e-7), (
            "SGD with weight_decay=0 and σ=0 should be bit-identical "
            "across two seeded runs."
        )
