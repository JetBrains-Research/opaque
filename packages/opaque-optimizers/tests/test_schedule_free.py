"""Tests for opaque.optimizers.schedule_free wrapper."""

from __future__ import annotations

import pytest
import torch

from opaque.optimizers import adamw, apply_updates, schedule_free, sgd
from opaque.optimizers.types import ScheduleFreeState


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
        _step, state = schedule_free(params, adamw, lr=1e-3)
        assert isinstance(state, ScheduleFreeState)
        for k in params:
            torch.testing.assert_close(state.z[k], params[k])
            torch.testing.assert_close(state.x[k], params[k])
        assert state.step == 0

    def test_eval_params_helper(self, params):
        _step, state = schedule_free(params, adamw, lr=1e-3)
        ep = state.x
        for k in params:
            torch.testing.assert_close(ep[k], state.x[k])

    def test_update_advances_step_and_x(self, params, grads):
        step, state = schedule_free(params, adamw, lr=1e-2, beta=0.9)
        _delta, state = step(grads, state, params=params)
        assert state.step == 1
        assert any(not torch.equal(state.x[k], params[k]) for k in params)

    def test_apply_updates_yields_consistent_y(self, params, grads):
        """Applying delta to params (=y_t) should produce y_{t+1}."""
        step, state = schedule_free(params, adamw, lr=1e-2, beta=0.9)
        delta, state = step(grads, state, params=params)
        new_y = apply_updates(params, delta)
        for k in params:
            expected = (1 - 0.9) * state.z[k] + 0.9 * state.x[k]
            torch.testing.assert_close(new_y[k], expected, atol=1e-6, rtol=1e-5)

    def test_warmup_x_tracks_z(self, params, grads):
        """During warmup, x should equal z (no averaging yet)."""
        step, state = schedule_free(params, adamw, lr=1e-2, warmup_steps=5)
        p = params
        for _ in range(3):
            delta, state = step(grads, state, params=p)
            p = apply_updates(p, delta)
        for k in state.x:
            torch.testing.assert_close(state.x[k], state.z[k])

    def test_requires_params(self, params, grads):
        step, state = schedule_free(params, adamw, lr=1e-3)
        with pytest.raises(TypeError, match="params"):
            step(grads, state)  # type: ignore[misc]

    def test_validation(self, params):
        with pytest.raises(ValueError, match="beta"):
            schedule_free(params, adamw, lr=1e-3, beta=1.5)
        with pytest.raises(ValueError, match="warmup"):
            schedule_free(params, adamw, lr=1e-3, warmup_steps=-1)

    def test_compatible_with_engine_sgd(self, params, grads):
        """Wrapper accepts engine optimizer factories."""
        step, state = schedule_free(params, sgd, lr=1e-2)
        delta, state = step(grads, state, params=params)
        assert state.step == 1
        for k in params:
            assert torch.isfinite(delta[k]).all()

    def test_weight_decay_references_z_not_y(self, params, grads):
        """Decoupled weight decay must regularise the raw iterate ``z``."""
        step, state = schedule_free(params, adamw, lr=0.1, weight_decay=0.5, beta=0.9)
        # Step once with non-zero grads so z, x, y diverge.
        delta, state = step(grads, state, params=params)
        params_after_first = apply_updates(params, delta)
        zero_grads = {k: torch.zeros_like(v) for k, v in params.items()}
        z_before = {k: v.clone() for k, v in state.z.items()}
        _, state = step(zero_grads, state, params=params_after_first)
        for k in params:
            shift = state.z[k] - z_before[k]
            z_dir = z_before[k]
            y_dir = params_after_first[k]
            cos_z = torch.nn.functional.cosine_similarity(
                shift.flatten().unsqueeze(0),
                z_dir.flatten().unsqueeze(0),
            ).item()
            cos_y = torch.nn.functional.cosine_similarity(
                shift.flatten().unsqueeze(0),
                y_dir.flatten().unsqueeze(0),
            ).item()
            assert cos_z > cos_y - 1e-6, (
                f"WD term aligns with y instead of z for {k!r} "
                f"(cos_z={cos_z:.4f}, cos_y={cos_y:.4f})"
            )

    def test_post_warmup_averaging_starts_fresh(self, params, grads):
        """After warmup ends, ``x`` should start a fresh average."""
        warmup = 5
        step, state = schedule_free(
            params, adamw, lr=1e-2, beta=0.9, warmup_steps=warmup
        )
        p = params
        for _ in range(warmup):
            delta, state = step(grads, state, params=p)
            p = apply_updates(p, delta)
            for k in state.x:
                torch.testing.assert_close(state.x[k], state.z[k])
        # First post-warmup step: x should equal the new z (fresh average).
        delta, state = step(grads, state, params=p)
        for k in state.x:
            torch.testing.assert_close(state.x[k], state.z[k])
