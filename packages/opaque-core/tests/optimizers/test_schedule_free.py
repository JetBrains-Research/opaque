"""Tests for opaque.optimizers.schedule_free wrapper."""

from __future__ import annotations

import pytest
import torch

torchopt = pytest.importorskip("torchopt")

from opaque.optimizers import (  # noqa: E402
    ScheduleFreeState,
    adamw,
    get_eval_params,
    schedule_free,
)


@pytest.fixture
def params():
    torch.manual_seed(0)
    return {"weight": torch.randn(4, 3), "bias": torch.randn(3)}


@pytest.fixture
def grads(params):
    torch.manual_seed(1)
    return {k: torch.randn_like(v) for k, v in params.items()}


class TestScheduleFreeWrapper:
    def test_state_carries_z_x_inner(self, params):
        opt = schedule_free(adamw(lr=1e-3))
        state = opt.init(params)
        assert isinstance(state, ScheduleFreeState)
        for k in params:
            torch.testing.assert_close(state.z[k], params[k])
            torch.testing.assert_close(state.x[k], params[k])
        assert state.step == 0

    def test_eval_params_helper(self, params):
        opt = schedule_free(adamw(lr=1e-3))
        state = opt.init(params)
        ep = get_eval_params(state)
        for k in params:
            torch.testing.assert_close(ep[k], state.x[k])

    def test_update_advances_step_and_x(self, params, grads):
        opt = schedule_free(adamw(lr=1e-2), beta=0.9)
        state = opt.init(params)
        delta, state = opt.update(grads, state, params=params)
        assert state.step == 1
        # x should differ from initial params (it's now an average of z,
        # which moved away from y₀ = z₀ = x₀ via the inner update).
        assert any(not torch.equal(state.x[k], params[k]) for k in params)

    def test_apply_updates_yields_consistent_y(self, params, grads):
        """Applying delta to params (=y_t) should produce y_{t+1}."""
        opt = schedule_free(adamw(lr=1e-2), beta=0.9)
        state = opt.init(params)
        delta, state = opt.update(grads, state, params=params)
        new_y = torchopt.apply_updates(params, delta)
        # By construction y_{t+1} = (1-β) z_{t+1} + β x_{t+1}.
        for k in params:
            expected = (1 - 0.9) * state.z[k] + 0.9 * state.x[k]
            torch.testing.assert_close(new_y[k], expected, atol=1e-6, rtol=1e-5)

    def test_warmup_x_tracks_z(self, params, grads):
        """During warmup, x should equal z (no averaging yet)."""
        opt = schedule_free(adamw(lr=1e-2), warmup_steps=5)
        state = opt.init(params)
        for _ in range(3):
            delta, state = opt.update(grads, state, params=params)
            params = torchopt.apply_updates(params, delta)
        for k in state.x:
            torch.testing.assert_close(state.x[k], state.z[k])

    def test_requires_params(self, params, grads):
        opt = schedule_free(adamw(lr=1e-3))
        state = opt.init(params)
        with pytest.raises(ValueError, match="params"):
            opt.update(grads, state)

    def test_validation(self):
        with pytest.raises(ValueError, match="beta"):
            schedule_free(adamw(lr=1e-3), beta=1.5)
        with pytest.raises(ValueError, match="warmup"):
            schedule_free(adamw(lr=1e-3), warmup_steps=-1)

    def test_compatible_with_torchopt_sgd(self, params, grads):
        """Wrapper accepts non-opaque base optimizers."""
        opt = schedule_free(torchopt.sgd(lr=1e-2))
        state = opt.init(params)
        delta, state = opt.update(grads, state, params=params)
        assert state.step == 1
        for k in params:
            assert torch.isfinite(delta[k]).all()
