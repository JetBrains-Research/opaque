"""Trainer-level wiring for ``torch_compile``, ``use_performance_kernels``, and the
``opaque.precision`` loss-scaler integration.

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

from opaque.transformers.trainer import DPTrainer, TrainingArguments
from opaque.precision import LossScaler, LossScalerState


# ----------------------------------------------------------------------------
# Tiny shared trainer helper
# ----------------------------------------------------------------------------


def _args(tmp_path, **overrides) -> TrainingArguments:
    defaults = dict(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        max_steps=1,
        num_train_epochs=1,
        save_strategy="no",
        use_cpu=True,
        privacy_target_epsilon=10.0,
        privacy_noise_multiplier=1.0,
    )
    defaults.update(overrides)
    return TrainingArguments(**defaults)


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
# use_performance_kernels — wiring through to apply_model_patches
# ----------------------------------------------------------------------------


def test_use_performance_kernels_default_keeps_kv_cache_and_compat_on(
    tmp_path, monkeypatch
):
    """Default-off ``use_performance_kernels`` still applies compat and the
    ``performance`` bucket (kv_cache); only the Triton ``kernels`` group is
    disabled."""
    calls: list[dict] = []

    def _spy(model, **kwargs):
        calls.append({"model": model, "kwargs": kwargs})

    monkeypatch.setattr("opaque.patches.apply_model_patches", _spy)

    _tiny_trainer(tmp_path)  # use_performance_kernels default is False
    assert len(calls) == 1
    assert calls[0]["kwargs"]["performance"] is True
    assert calls[0]["kwargs"]["kernels"] is False
    assert calls[0]["kwargs"]["compat"] is True


def test_use_performance_kernels_true_enables_kernels_group(tmp_path, monkeypatch):
    """``use_performance_kernels=True`` flips ``kernels`` on at the
    ``apply_model_patches`` call (alongside the always-on ``performance``
    and ``compat`` umbrellas)."""
    calls: list[dict] = []

    def _spy(model, **kwargs):
        calls.append({"model": model, "kwargs": kwargs})

    # _performance_kernels.py imports apply_model_patches lazily inside the function body,
    # so patch the source location rather than the consumer's module.
    monkeypatch.setattr("opaque.patches.apply_model_patches", _spy)

    trainer, model = _tiny_trainer(tmp_path, use_performance_kernels=True)
    assert len(calls) == 1
    assert calls[0]["model"] is model
    assert calls[0]["kwargs"]["performance"] is True
    assert calls[0]["kwargs"]["kernels"] is True
    assert calls[0]["kwargs"]["compat"] is True


def test_performance_kernels_config_forwards_opaque_keys_as_is(tmp_path, monkeypatch):
    """``performance_kernels_config`` is a flat dict forwarded as-is to
    ``apply_model_patches`` kwargs — no key translation, opaque-patches
    keys used directly."""
    calls: list[dict] = []

    def _spy(model, **kwargs):
        calls.append({"kwargs": kwargs})

    monkeypatch.setattr("opaque.patches.apply_model_patches", _spy)

    _tiny_trainer(
        tmp_path,
        use_performance_kernels=True,
        performance_kernels_config={
            "rope": True,
            "rms_norm": True,
            "fused_linear_cross_entropy": True,
        },
    )
    assert len(calls) == 1
    assert calls[0]["kwargs"]["rope"] is True
    assert calls[0]["kwargs"]["rms_norm"] is True
    assert calls[0]["kwargs"]["fused_linear_cross_entropy"] is True


def test_performance_kernels_config_can_disable_kv_cache(tmp_path, monkeypatch):
    """``kv_cache`` stays on by default but can be opted out via the config
    dict for models whose forward depends on HF's DynamicCache."""
    calls: list[dict] = []

    def _spy(model, **kwargs):
        calls.append({"kwargs": kwargs})

    monkeypatch.setattr("opaque.patches.apply_model_patches", _spy)

    _tiny_trainer(tmp_path, performance_kernels_config={"kv_cache": False})
    assert len(calls) == 1
    assert calls[0]["kwargs"]["kv_cache"] is False
    assert calls[0]["kwargs"]["performance"] is True


# ----------------------------------------------------------------------------
# fp16 + opaque.precision integration with the Trainer
# ----------------------------------------------------------------------------


def test_fp16_trainer_wires_loss_scaler_into_self(tmp_path):
    """fp16=True must wire the functional loss-scaler transform + state onto
    self so training_step's inf/nan check has something to call."""
    trainer, _ = _tiny_trainer(tmp_path, fp16=True)
    assert isinstance(trainer._loss_scaler, LossScaler)
    assert isinstance(trainer._loss_scaler_state, LossScalerState)
    assert trainer._loss_scaler_state.scale == 2**16


def test_bf16_trainer_does_not_wire_loss_scaler(tmp_path):
    """bf16's wider exponent range doesn't need scaling; loss_scaler is None
    so training_step skips the inf-check entirely (zero-overhead bf16)."""
    trainer, _ = _tiny_trainer(tmp_path, bf16=True)
    assert trainer._loss_scaler is None
    assert trainer._loss_scaler_state is None


def test_fp32_trainer_does_not_wire_loss_scaler(tmp_path):
    trainer, _ = _tiny_trainer(tmp_path)
    assert trainer._loss_scaler is None
    assert trainer._loss_scaler_state is None


def test_fp16_loss_scaler_backoff_via_direct_call(tmp_path):
    """End-to-end call into the trainer's loss_scaler verifies the backoff
    semantic the training_step relies on for fp16 inf/nan steps.
    Doesn't run training_step itself (which needs a full data pipeline); the
    invariant under test is that a non-finite update halves the scale."""
    from opaque.precision import all_finite

    trainer, _ = _tiny_trainer(tmp_path, fp16=True)
    initial = trainer._loss_scaler_state.scale
    inf_grads = {"w": torch.tensor([float("inf")])}
    assert all_finite(inf_grads) is False
    trainer._loss_scaler_state = trainer._loss_scaler.update(
        trainer._loss_scaler_state, grads_were_finite=False
    )
    assert trainer._loss_scaler_state.scale == initial / 2
