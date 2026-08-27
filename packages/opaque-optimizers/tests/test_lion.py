"""Tests for opaque.optimizers._lion."""

from __future__ import annotations

import pytest
import torch

from opaque.optimizers import apply_updates, lion
from opaque.optimizers.types import LionState
from opaque.types import (
    clipped,
    noised,
)


@pytest.fixture
def params():
    torch.manual_seed(0)
    return {"weight": torch.randn(4, 3), "bias": torch.randn(3)}


@pytest.fixture
def grads(params):
    torch.manual_seed(1)
    return {k: torch.randn_like(v) for k, v in params.items()}


def _lion_state(state: LionState) -> LionState:
    return state


class TestLion:
    def test_factory_returns_step_and_state(self, params):
        step, state = lion(params, lr=1e-4)
        assert callable(step)
        assert isinstance(state, LionState)

    def test_state_has_only_m(self, params):
        _, state = lion(params, lr=1e-4)
        ls = _lion_state(state)
        assert isinstance(ls, LionState)
        assert ls.step == 0
        for k in params:
            assert torch.equal(ls.m[k], torch.zeros_like(params[k]))

    def test_update_is_signed(self, params, grads):
        """Lion's update direction is the sign of (β₁m + (1-β₁)g)."""
        step, state = lion(
            params, lr=1.0, weight_decay=0.0
        )  # lr=1 so the sign survives.
        updates, _ = step(grads, state, params=params)
        for k in updates:
            # update = -lr * sign(...), so all entries are ±lr.
            mags = updates[k].abs()
            torch.testing.assert_close(mags, torch.ones_like(mags))

    def test_step_increments(self, params, grads):
        step, state = lion(params, lr=1e-4)
        _, state = step(grads, state, params=params)
        assert _lion_state(state).step == 1
        _, state = step(grads, state, params=params)
        assert _lion_state(state).step == 2

    def test_apply_updates_changes_params(self, params, grads):
        step, state = lion(params, lr=1e-2)
        orig = {k: v.clone() for k, v in params.items()}
        updates, _ = step(grads, state, params=params)
        new = apply_updates(params, updates)
        changed = any(not torch.equal(new[k], orig[k]) for k in params)
        assert changed

    def test_decoupled_wd_with_zero_grad(self):
        params = {"w": torch.ones(3) * 2.0}
        grads = {"w": torch.zeros(3)}
        step, state = lion(
            params, lr=0.1, weight_decay=0.5, decoupled_weight_decay=True
        )
        updates, _ = step(grads, state, params=params)
        # sign(0) = 0 → only WD survives.  update = -lr * wd * params.
        expected = -0.1 * 0.5 * params["w"]
        torch.testing.assert_close(updates["w"], expected)

    def test_validation(self):
        with pytest.raises(ValueError, match="beta_1"):
            lion({"w": torch.ones(1)}, betas=(1.0, 0.99))
        with pytest.raises(ValueError, match="weight_decay must be non-negative"):
            lion({"w": torch.ones(1)}, weight_decay=-0.1)

    def test_dp_kwargs_rejected(self, params, grads):
        """Lion has no second moment, so DP-aware kwargs aren't part of
        its update signature; passing one raises ``TypeError`` instead
        of silently ignoring it (which would mislead users into thinking
        DP-BC was active when it wasn't)."""
        step, state = lion(params, lr=1e-4)
        with pytest.raises(TypeError, match="noise_stddev"):
            step(grads, state, params=params, noise_stddev=0.5)
        with pytest.raises(TypeError, match="noisy_squared_grads"):
            step(grads, state, params=params, noisy_squared_grads={})

    def test_noisy_updates_unwrap_silently(self, params, grads):
        """Lion has no DP-aware path, so it silently uses the wrapped
        pytree values when handed a NoisedPytree.  No warning."""
        step, state = lion(params, lr=1e-4)
        updates, _ = step(
            noised(grads, max_norm=1.0, noise_stddev=0.25),
            state,
            params=params,
        )
        for k in params:
            assert updates[k].shape == params[k].shape

    def test_clipped_updates_are_rejected(self, params, grads):
        step, state = lion(params, lr=1e-4)
        with pytest.raises(
            TypeError, match="have not passed through a noise mechanism"
        ):
            step(clipped(grads, max_norm=1.0), state, params=params)
