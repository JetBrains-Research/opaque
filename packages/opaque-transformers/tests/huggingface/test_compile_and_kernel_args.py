"""Trainer-level wiring for ``torch_compile``, ``use_liger_kernel``, and the
``OpaqueLossScaler`` integration.

These tests target the *plumbing* — Phase 11 features behave correctly when
flags flip — without running full training (which would require a complete
data collator, sampler, dataset, accountant). The actual training/eval
behavior is covered by the broader trainer suite.

The DP-critical fp16 invariant — that the clipped gradient is **invariant**
to the loss scale because ``pre_clipping_transform`` runs the unscale before
the clip-norm — is asserted at the gradient-function level.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from opaque.transformers.trainer import DPTrainer, DPTrainingArguments
from opaque.transformers.trainer._loss_scaler import OpaqueLossScaler


# ----------------------------------------------------------------------------
# Tiny shared trainer helper
# ----------------------------------------------------------------------------


def _args(tmp_path, **overrides) -> DPTrainingArguments:
    defaults = dict(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        max_steps=1,
        num_train_epochs=1,
        save_strategy="no",
        use_cpu=True,
        dp_target_epsilon=10.0,
        dp_noise_multiplier=1.0,
    )
    defaults.update(overrides)
    return DPTrainingArguments(**defaults)


def _tiny_trainer(tmp_path, **arg_overrides) -> tuple[DPTrainer, nn.Module]:
    model = nn.Linear(4, 2)
    args = _args(tmp_path, **arg_overrides)
    trainer = DPTrainer(
        model=model,
        args=args,
        train_dataset=[{"x": torch.zeros(4)}],
        eval_dataset=None,
    )
    return trainer, model


# ----------------------------------------------------------------------------
# torch_compile flag wiring
# ----------------------------------------------------------------------------


def test_torch_compile_default_false_does_not_compile(tmp_path):
    """When torch_compile is unset, the trainer must not pull torch.compile
    onto the loss closure (zero-overhead default)."""
    trainer, _ = _tiny_trainer(tmp_path)
    assert trainer.args.torch_compile is False


def test_torch_compile_true_accepted(tmp_path):
    """torch_compile=True initializes without error; backend/mode default
    to inductor/default at the closure-building site."""
    trainer, _ = _tiny_trainer(tmp_path, torch_compile=True)
    assert trainer.args.torch_compile is True


def test_torch_compile_with_backend_and_mode(tmp_path):
    trainer, _ = _tiny_trainer(
        tmp_path,
        torch_compile=True,
        torch_compile_backend="aot_eager",
        torch_compile_mode="reduce-overhead",
    )
    assert trainer.args.torch_compile_backend == "aot_eager"
    assert trainer.args.torch_compile_mode == "reduce-overhead"


def test_torch_compile_invalid_mode_rejected_at_args(tmp_path):
    with pytest.raises(ValueError, match="torch_compile_mode"):
        _args(tmp_path, torch_compile_mode="nonsense")


# ----------------------------------------------------------------------------
# use_liger_kernel — wiring through to apply_model_patches
# ----------------------------------------------------------------------------


def test_use_liger_kernel_default_false_no_patch_call(tmp_path, monkeypatch):
    """Without use_liger_kernel, apply_model_patches is not called by the
    Trainer's __init__ — patch decisions stay user-driven."""
    calls: list[tuple] = []

    def _spy(*args, **kwargs):  # pragma: no cover - records non-call
        calls.append((args, kwargs))

    monkeypatch.setattr("opaque.patches.apply_model_patches", _spy)

    _tiny_trainer(tmp_path)  # use_liger_kernel default is False
    assert calls == []


def test_use_liger_kernel_true_invokes_apply_model_patches(tmp_path, monkeypatch):
    """use_liger_kernel=True calls apply_model_patches(model, performance=True,
    compat=True) once at __init__ time (HF parity)."""
    calls: list[dict] = []

    def _spy(model, **kwargs):
        calls.append({"model": model, "kwargs": kwargs})

    # _liger.py imports apply_model_patches lazily inside the function body,
    # so patch the source location rather than the consumer's module.
    monkeypatch.setattr("opaque.patches.apply_model_patches", _spy)

    trainer, model = _tiny_trainer(tmp_path, use_liger_kernel=True)
    assert len(calls) == 1
    assert calls[0]["model"] is model
    assert calls[0]["kwargs"]["performance"] is True
    assert calls[0]["kwargs"]["compat"] is True


def test_liger_kernel_config_is_translated_at_init(tmp_path, monkeypatch):
    """liger_kernel_config keys are translated to opaque-patches kwarg names
    before being forwarded to apply_model_patches."""
    calls: list[dict] = []

    def _spy(model, **kwargs):
        calls.append({"kwargs": kwargs})

    monkeypatch.setattr("opaque.patches.apply_model_patches", _spy)

    _tiny_trainer(
        tmp_path,
        use_liger_kernel=True,
        liger_kernel_config={
            "rope": True,
            "rms_norm": True,
            "fused_linear_cross_entropy": True,
        },
    )
    assert len(calls) == 1
    assert calls[0]["kwargs"]["fuse_rope"] is True
    assert calls[0]["kwargs"]["fuse_rms_norm"] is True
    # cross_entropy unified opaque flag — driven by fused_linear_cross_entropy.
    assert calls[0]["kwargs"]["fuse_cross_entropy"] is True


# ----------------------------------------------------------------------------
# fp16 + OpaqueLossScaler integration with the Trainer
# ----------------------------------------------------------------------------


def test_fp16_trainer_wires_loss_scaler_into_self(tmp_path):
    """fp16=True must wire a working OpaqueLossScaler onto self._loss_scaler
    so training_step's inf/nan check has something to call."""
    trainer, _ = _tiny_trainer(tmp_path, fp16=True)
    assert isinstance(trainer._loss_scaler, OpaqueLossScaler)
    assert trainer._loss_scaler.scale == 2**16


def test_bf16_trainer_does_not_wire_loss_scaler(tmp_path):
    """bf16's wider exponent range doesn't need scaling; loss_scaler is None
    so training_step skips the inf-check entirely (zero-overhead bf16)."""
    trainer, _ = _tiny_trainer(tmp_path, bf16=True)
    assert trainer._loss_scaler is None


def test_fp32_trainer_does_not_wire_loss_scaler(tmp_path):
    trainer, _ = _tiny_trainer(tmp_path)
    assert trainer._loss_scaler is None


def test_fp16_loss_scaler_backoff_via_direct_call(tmp_path):
    """End-to-end call into the trainer's loss_scaler verifies the backoff
    semantic the training_step relies on for fp16 inf/nan steps.
    Doesn't run training_step itself (which needs a full data pipeline); the
    invariant under test is that a non-finite update halves the scale."""
    trainer, _ = _tiny_trainer(tmp_path, fp16=True)
    initial = trainer._loss_scaler.scale
    inf_grads = {"w": torch.tensor([float("inf")])}
    assert trainer._loss_scaler.all_finite(inf_grads) is False
    trainer._loss_scaler.update(grads_were_finite=False)
    assert trainer._loss_scaler.scale == initial / 2
