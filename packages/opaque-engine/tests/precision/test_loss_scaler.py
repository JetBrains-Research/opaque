"""Unit tests for :func:`opaque.precision.loss_scaler`.

The functional analog of :class:`torch.amp.GradScaler` for DP-SGD's pytree
gradient flow. These tests pin down the four invariants DP correctness and
HF-parity rely on:

1. ``scale_loss`` multiplies; ``unscale_grads`` is its left-inverse.
2. ``all_finite`` is True iff every floating-point leaf is finite.
3. ``update`` mirrors ``GradScaler``'s growth/backoff state machine.
4. Roundtrip (scale → grad → unscale) is a no-op on the gradient pytree.

For the *DP-critical* invariant (unscale runs before clipping so the
accountant sees true sensitivity) see
``packages/opaque-dpsgd/tests/functional/test_precision.py`` and
``packages/opaque-transformers/tests/opaque_transformers/test_precision_training.py``.
"""

from __future__ import annotations

import pytest
import torch

from opaque.api.engine.clipping import auto_clipped_grad
from opaque.precision import LossScaler, LossScalerState, all_finite, loss_scaler
from opaque.types import NoisedPytree, SecondMomentClippingOutput, clipped


def _grads_pytree(magnitude: float = 1.0) -> dict[str, torch.Tensor]:
    """Realistic pytree: dict of fp32 + bf16 + a non-floating tensor."""
    return {
        "fp32": torch.full((3, 3), magnitude, dtype=torch.float32),
        "bf16": torch.full((4,), magnitude, dtype=torch.bfloat16),
        "step": torch.tensor(7, dtype=torch.int64),  # passthrough
    }


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------


def test_factory_returns_transform_and_state():
    scaler, state = loss_scaler()
    assert isinstance(scaler, LossScaler)
    assert isinstance(state, LossScalerState)


def test_factory_default_scale_matches_grad_scaler():
    _, state = loss_scaler()
    assert state.scale == 2**16  # torch.amp.GradScaler default
    assert state.growth_tracker == 0


def test_factory_custom_init_scale():
    _, state = loss_scaler(init_scale=128.0)
    assert state.scale == 128.0


def test_factory_validates_growth_factor():
    with pytest.raises(ValueError, match="growth_factor"):
        loss_scaler(growth_factor=1.0)


def test_factory_validates_backoff_factor():
    with pytest.raises(ValueError, match="backoff_factor"):
        loss_scaler(backoff_factor=0.0)
    with pytest.raises(ValueError, match="backoff_factor"):
        loss_scaler(backoff_factor=1.0)


def test_factory_validates_growth_interval():
    with pytest.raises(ValueError, match="growth_interval"):
        loss_scaler(growth_interval=0)


# ----------------------------------------------------------------------------
# scale_loss / unscale_grads roundtrip
# ----------------------------------------------------------------------------


def test_scale_loss_multiplies():
    scaler, state = loss_scaler(init_scale=128.0)
    loss = torch.tensor(0.5)
    assert scaler.scale_loss(loss, state).item() == pytest.approx(64.0)


def test_unscale_grads_preserves_pytree_structure():
    scaler, state = loss_scaler(init_scale=128.0)
    grads = _grads_pytree(magnitude=128.0)
    unscaled = scaler.unscale_grads(grads, state)
    assert set(unscaled) == set(grads)
    assert unscaled["fp32"].dtype == torch.float32
    assert unscaled["bf16"].dtype == torch.bfloat16
    assert torch.equal(unscaled["step"], grads["step"])  # int passthrough


def test_unscale_recovers_original_magnitude():
    scaler, state = loss_scaler(init_scale=128.0)
    original = _grads_pytree(magnitude=1.0)
    scaled = {
        k: (v * 128.0 if v.is_floating_point() else v) for k, v in original.items()
    }
    unscaled = scaler.unscale_grads(scaled, state)
    torch.testing.assert_close(unscaled["fp32"], original["fp32"])
    torch.testing.assert_close(unscaled["bf16"], original["bf16"])


# ----------------------------------------------------------------------------
# all_finite (free function)
# ----------------------------------------------------------------------------


def test_all_finite_true_on_normal_pytree():
    assert all_finite(_grads_pytree(magnitude=1.0)) is True


def test_all_finite_false_when_any_leaf_has_inf():
    grads = _grads_pytree()
    grads["fp32"][0, 0] = float("inf")
    assert all_finite(grads) is False


def test_all_finite_false_when_any_leaf_has_nan():
    grads = _grads_pytree()
    grads["bf16"][2] = float("nan")
    assert all_finite(grads) is False


def test_all_finite_ignores_integer_leaves():
    """Integer leaves can't be inf/nan; passthrough tensors must not break detection."""
    grads = {
        "ok": torch.zeros(2),
        "step": torch.tensor(2**60, dtype=torch.int64),  # legitimately huge int
    }
    assert all_finite(grads) is True


def test_all_finite_false_for_clipped_pytree_wrapper():
    grads = clipped({"w": torch.tensor([1.0, float("nan")])}, max_norm=1.0)
    assert all_finite(grads) is False


def test_all_finite_false_for_noised_pytree_wrapper():
    grads = NoisedPytree(
        pytree={"w": torch.tensor([1.0, float("inf")])},
        max_norm=1.0,
        noise_stddev=0.5,
    )
    assert all_finite(grads) is False


def test_all_finite_false_for_second_moment_clipping_output():
    grads = SecondMomentClippingOutput(
        grads=clipped({"w": torch.tensor([1.0])}, max_norm=1.0),
        squared_grads=clipped({"w": torch.tensor([float("nan")])}, max_norm=1.0),
    )
    assert all_finite(grads) is False


def test_all_finite_false_for_wrapped_clipped_pytree_inside_plain_pytree():
    grads = {
        "outer": clipped({"w": torch.tensor([1.0, float("nan")])}, max_norm=1.0),
        "step": torch.tensor(1, dtype=torch.int64),
    }
    assert all_finite(grads) is False


def test_all_finite_false_for_auto_clipped_grad_wrapper_output():
    def loss_fn(param, data):
        return torch.sqrt(param - data).mean()

    grad_fn, state = auto_clipped_grad(loss_fn, argnums=0, batch_argnums=1, R=1.0)
    grads, _ = grad_fn(
        torch.tensor(1.0, requires_grad=True),
        torch.tensor([0.5, 2.0]),
        state=state,
    )
    grads.pytree.fill_(float("nan"))
    assert all_finite(grads) is False


# ----------------------------------------------------------------------------
# update — growth/backoff state machine
# ----------------------------------------------------------------------------


def test_update_grows_after_growth_interval_clean_steps():
    scaler, state = loss_scaler(init_scale=128.0, growth_factor=2.0, growth_interval=3)
    for _ in range(2):
        state = scaler.update(state, grads_were_finite=True)
        assert state.scale == 128.0  # not yet
    state = scaler.update(state, grads_were_finite=True)
    assert state.scale == 256.0  # third clean step grows


def test_update_backs_off_on_inf():
    scaler, state = loss_scaler(init_scale=128.0, backoff_factor=0.5)
    state = scaler.update(state, grads_were_finite=False)
    assert state.scale == 64.0


def test_update_backoff_resets_growth_tracker():
    scaler, state = loss_scaler(init_scale=128.0, growth_interval=3)
    for _ in range(2):  # two clean steps
        state = scaler.update(state, grads_were_finite=True)
    state = scaler.update(state, grads_were_finite=False)  # backoff resets tracker
    assert state.scale == 64.0
    # After backoff, tracker is reset; next two clean steps shouldn't grow.
    for _ in range(2):
        state = scaler.update(state, grads_were_finite=True)
    assert state.scale == 64.0  # third clean step would grow, but we stop at 2


def test_update_returns_new_state_immutable():
    """``LossScalerState`` is frozen — ``update`` must return a fresh instance."""
    scaler, state = loss_scaler(init_scale=128.0, growth_interval=1)
    new_state = scaler.update(state, grads_were_finite=True)
    assert new_state is not state
    assert state.scale == 128.0  # original untouched


# ----------------------------------------------------------------------------
# enabled=False — pure passthrough
# ----------------------------------------------------------------------------


def test_disabled_scale_loss_is_identity():
    scaler, state = loss_scaler(enabled=False)
    loss = torch.tensor(0.5)
    assert scaler.scale_loss(loss, state) is loss


def test_disabled_unscale_grads_is_identity():
    scaler, state = loss_scaler(enabled=False)
    grads = _grads_pytree()
    assert scaler.unscale_grads(grads, state) is grads


def test_disabled_update_is_noop():
    scaler, state = loss_scaler(enabled=False, init_scale=128.0)
    new_state = scaler.update(state, grads_were_finite=False)
    assert new_state is state  # unchanged, same instance
    assert new_state.scale == 128.0


# ----------------------------------------------------------------------------
# State serialization (via opaque.serialization)
# ----------------------------------------------------------------------------


def test_state_round_trips_through_opaque_serialization():
    from opaque.serialization import from_state_dict, state_dict

    scaler, state = loss_scaler(init_scale=128.0, growth_interval=2)
    state = scaler.update(state, grads_were_finite=True)
    state = scaler.update(state, grads_were_finite=True)  # second clean → grow
    payload = state_dict(state)
    template = LossScalerState(scale=0.0, growth_tracker=0)
    restored = from_state_dict(template, payload)
    assert restored.scale == state.scale
    assert restored.growth_tracker == state.growth_tracker
