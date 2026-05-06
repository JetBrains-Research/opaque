"""Unit tests for ``opaque.transformers.trainer._loss_scaler.OpaqueLossScaler``.

The functional analog of ``torch.amp.GradScaler`` for DP-SGD's pytree-based
gradient flow.  These tests pin down the four invariants that DP correctness
and HF-parity rely on:

1. ``scale_loss`` multiplies; ``unscale_grads`` is its left-inverse.
2. ``all_finite`` is True iff every floating-point leaf is finite.
3. ``update`` mirrors ``GradScaler``'s growth/backoff state machine.
4. Roundtrip (scale → grad → unscale) is a no-op on the gradient pytree.

For the *DP-critical* invariant (unscale runs before clipping so the
accountant sees true sensitivity), see ``test_precision_training.py``.
"""

from __future__ import annotations

import math

import pytest
import torch

from opaque.transformers.trainer._loss_scaler import OpaqueLossScaler


def _grads_pytree(magnitude: float = 1.0) -> dict[str, torch.Tensor]:
    """Realistic pytree: dict of fp32 + bf16 + a non-floating tensor."""
    return {
        "fp32": torch.full((3, 3), magnitude, dtype=torch.float32),
        "bf16": torch.full((4,), magnitude, dtype=torch.bfloat16),
        "step": torch.tensor(7, dtype=torch.int64),  # passthrough
    }


# ----------------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------------


def test_init_default_scale_matches_grad_scaler():
    scaler = OpaqueLossScaler()
    assert scaler.scale == 2**16  # torch.amp.GradScaler default


def test_init_validates_growth_factor():
    with pytest.raises(ValueError, match="growth_factor"):
        OpaqueLossScaler(growth_factor=1.0)


def test_init_validates_backoff_factor():
    with pytest.raises(ValueError, match="backoff_factor"):
        OpaqueLossScaler(backoff_factor=0.0)
    with pytest.raises(ValueError, match="backoff_factor"):
        OpaqueLossScaler(backoff_factor=1.0)


def test_init_validates_growth_interval():
    with pytest.raises(ValueError, match="growth_interval"):
        OpaqueLossScaler(growth_interval=0)


# ----------------------------------------------------------------------------
# scale_loss / unscale_grads roundtrip
# ----------------------------------------------------------------------------


def test_scale_loss_multiplies():
    scaler = OpaqueLossScaler(init_scale=128.0)
    loss = torch.tensor(0.5)
    assert scaler.scale_loss(loss).item() == pytest.approx(64.0)


def test_unscale_grads_preserves_pytree_structure():
    scaler = OpaqueLossScaler(init_scale=128.0)
    grads = _grads_pytree(magnitude=128.0)
    unscaled = scaler.unscale_grads(grads)
    assert set(unscaled) == set(grads)
    assert unscaled["fp32"].dtype == torch.float32
    assert unscaled["bf16"].dtype == torch.bfloat16
    assert torch.equal(unscaled["step"], grads["step"])  # int passthrough


def test_unscale_recovers_original_magnitude():
    scaler = OpaqueLossScaler(init_scale=128.0)
    original = _grads_pytree(magnitude=1.0)
    scaled = {
        k: (v * 128.0 if v.is_floating_point() else v) for k, v in original.items()
    }
    unscaled = scaler.unscale_grads(scaled)
    torch.testing.assert_close(unscaled["fp32"], original["fp32"])
    torch.testing.assert_close(unscaled["bf16"], original["bf16"])


# ----------------------------------------------------------------------------
# all_finite
# ----------------------------------------------------------------------------


def test_all_finite_true_on_normal_pytree():
    scaler = OpaqueLossScaler()
    assert scaler.all_finite(_grads_pytree(magnitude=1.0)) is True


def test_all_finite_false_when_any_leaf_has_inf():
    scaler = OpaqueLossScaler()
    grads = _grads_pytree()
    grads["fp32"][0, 0] = float("inf")
    assert scaler.all_finite(grads) is False


def test_all_finite_false_when_any_leaf_has_nan():
    scaler = OpaqueLossScaler()
    grads = _grads_pytree()
    grads["bf16"][2] = float("nan")
    assert scaler.all_finite(grads) is False


def test_all_finite_ignores_integer_leaves():
    """Integer leaves can't be inf/nan; passthrough tensors must not break detection."""
    scaler = OpaqueLossScaler()
    grads = {
        "ok": torch.zeros(2),
        "step": torch.tensor(2**60, dtype=torch.int64),  # legitimately huge int
    }
    assert scaler.all_finite(grads) is True


# ----------------------------------------------------------------------------
# update — growth/backoff state machine
# ----------------------------------------------------------------------------


def test_update_grows_after_growth_interval_clean_steps():
    scaler = OpaqueLossScaler(init_scale=128.0, growth_factor=2.0, growth_interval=3)
    for _ in range(2):
        scaler.update(grads_were_finite=True)
        assert scaler.scale == 128.0  # not yet
    scaler.update(grads_were_finite=True)
    assert scaler.scale == 256.0  # third clean step grows


def test_update_backs_off_on_inf():
    scaler = OpaqueLossScaler(init_scale=128.0, backoff_factor=0.5)
    scaler.update(grads_were_finite=False)
    assert scaler.scale == 64.0


def test_update_backoff_resets_growth_tracker():
    scaler = OpaqueLossScaler(init_scale=128.0, growth_interval=3)
    for _ in range(2):  # two clean steps
        scaler.update(grads_were_finite=True)
    scaler.update(grads_were_finite=False)  # backoff resets tracker
    assert scaler.scale == 64.0
    # After backoff, tracker is reset; next two clean steps shouldn't grow.
    for _ in range(2):
        scaler.update(grads_were_finite=True)
    assert scaler.scale == 64.0  # third clean step would grow, but we stop at 2


# ----------------------------------------------------------------------------
# enabled=False — pure passthrough
# ----------------------------------------------------------------------------


def test_disabled_scale_loss_is_identity():
    scaler = OpaqueLossScaler(enabled=False)
    loss = torch.tensor(0.5)
    assert scaler.scale_loss(loss) is loss


def test_disabled_unscale_grads_is_identity():
    scaler = OpaqueLossScaler(enabled=False)
    grads = _grads_pytree()
    assert scaler.unscale_grads(grads) is grads


def test_disabled_all_finite_always_true():
    scaler = OpaqueLossScaler(enabled=False)
    grads = _grads_pytree()
    grads["fp32"][0, 0] = float("inf")
    assert scaler.all_finite(grads) is True


def test_disabled_update_is_noop():
    scaler = OpaqueLossScaler(enabled=False, init_scale=128.0)
    scaler.update(grads_were_finite=False)
    assert scaler.scale == 128.0  # unchanged


# ----------------------------------------------------------------------------
# state_dict roundtrip
# ----------------------------------------------------------------------------


def test_state_dict_roundtrip():
    a = OpaqueLossScaler(init_scale=64.0, growth_factor=4.0)
    a.update(grads_were_finite=False)  # mutate state
    a.update(grads_were_finite=True)
    state = a.state_dict()

    b = OpaqueLossScaler()  # different defaults
    b.load_state_dict(state)
    assert math.isclose(b.scale, a.scale)
    assert b.growth_factor == a.growth_factor
    assert b.init_scale == a.init_scale
    assert b._growth_tracker == a._growth_tracker
