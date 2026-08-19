"""Tests for opaque.optimizers._rmsprop."""

from __future__ import annotations

import pytest
import torch

from opaque.optimizers import apply_updates, rmsprop
from opaque.optimizers.types import RMSpropState
from opaque.types import (
    SecondMomentNoiseOutput,
    noised,
)

torchopt = pytest.importorskip("torchopt")


@pytest.fixture
def params():
    torch.manual_seed(0)
    return {"weight": torch.randn(4, 3), "bias": torch.randn(3)}


@pytest.fixture
def grads(params):
    torch.manual_seed(1)
    return {k: torch.randn_like(v) for k, v in params.items()}


def _rms_state(state: RMSpropState) -> RMSpropState:
    return state


class TestVanilla:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"lr": 1e-2, "alpha": 0.99, "eps": 1e-8, "weight_decay": 0.0},
            {"lr": 5e-3, "alpha": 0.95, "eps": 1e-6, "weight_decay": 0.0},
            {"lr": 0.1, "alpha": 0.9, "eps": 1e-4, "weight_decay": 0.0},
        ],
        ids=["default", "alpha_095", "high_lr"],
    )
    def test_matches_torchopt_rmsprop(self, params, kwargs):
        """Vanilla RMSprop is numerically identical to torchopt.rmsprop."""
        step, state = rmsprop(params, **kwargs)
        opt_ref = torchopt.rmsprop(**kwargs)
        state_ref = opt_ref.init(params)

        torch.manual_seed(42)
        for _ in range(10):
            step_grads = {k: torch.randn_like(v) for k, v in params.items()}
            updates, state = step(step_grads, state, params=params)
            updates_ref, state_ref = opt_ref.update(
                step_grads, state_ref, params=params
            )
            for k in params:
                torch.testing.assert_close(updates[k], updates_ref[k])

    def test_state_carries_phi_at_zero(self, params):
        _step, state = rmsprop(params, lr=1e-2)
        st = _rms_state(state)
        assert isinstance(st, RMSpropState)
        assert st.phi == 0.0
        assert st.step == 0

    def test_v_advances_via_ema(self, params, grads):
        alpha = 0.99
        step, state = rmsprop(params, lr=1e-2, alpha=alpha)
        _, state = step(grads, state, params=params)
        st = _rms_state(state)
        for k in grads:
            torch.testing.assert_close(st.nu[k], (1 - alpha) * grads[k] * grads[k])

    def test_step_increments(self, params, grads):
        step, state = rmsprop(params, lr=1e-2)
        _, state = step(grads, state, params=params)
        assert _rms_state(state).step == 1
        _, state = step(grads, state, params=params)
        assert _rms_state(state).step == 2

    def test_apply_updates_changes_params(self, params, grads):
        step, state = rmsprop(params, lr=1e-1)
        orig = {k: v.clone() for k, v in params.items()}
        updates, _ = step(grads, state, params=params)
        new = apply_updates(params, updates)
        assert any(not torch.equal(new[k], orig[k]) for k in params)


class TestBCMode:
    def test_phi_advances_under_noisy_metadata(self, params, grads):
        alpha = 0.99
        sigma = 0.5
        step, state = rmsprop(params, lr=1e-2, alpha=alpha, noise_bias_correction=True)
        expected_phi = 0.0
        for _ in range(10):
            _, state = step(
                noised(grads, max_norm=1.0, noise_stddev=sigma),
                state,
                params=params,
            )
            expected_phi = alpha * expected_phi + (1 - alpha) * (sigma**2)
        phi = _rms_state(state).phi
        assert isinstance(phi, dict)
        assert all(v == pytest.approx(expected_phi) for v in phi.values())

    def test_noisy_updates_take_per_step_metadata(self, params, grads):
        alpha = 0.99
        step, state = rmsprop(params, lr=1e-2, alpha=alpha, noise_bias_correction=True)
        expected_phi = 0.0
        for sigma in [0.1, 0.2, 0.3, 0.2, 0.1]:
            _, state = step(
                noised(grads, max_norm=1.0, noise_stddev=sigma),
                state,
                params=params,
            )
            expected_phi = alpha * expected_phi + (1 - alpha) * (sigma**2)
        phi = _rms_state(state).phi
        assert isinstance(phi, dict)
        assert all(v == pytest.approx(expected_phi) for v in phi.values())

    def test_bc_increases_effective_lr(self, params, grads):
        big = {k: v * 10 for k, v in grads.items()}
        step_std, s_std = rmsprop(params, lr=1e-2)
        step_bc, s_bc = rmsprop(params, lr=1e-2, noise_bias_correction=True)
        for _ in range(10):
            u_std, s_std = step_std(big, s_std, params=params)
            u_bc, s_bc = step_bc(
                noised(big, max_norm=1.0, noise_stddev=0.01),
                s_bc,
                params=params,
            )
        norm_std = sum(u.norm() for u in u_std.values())
        norm_bc = sum(u.norm() for u in u_bc.values())
        assert norm_bc >= norm_std

    def test_floor_prevents_zero_denominator(self):
        params = {"w": torch.ones(3)}
        grads = {"w": torch.ones(3) * 0.01}
        step, state = rmsprop(params, lr=1e-3, noise_bias_correction=True)
        updates, _ = step(
            noised(grads, max_norm=1.0, noise_stddev=1e6),
            state,
            params=params,
        )
        assert torch.isfinite(updates["w"]).all()

    def test_bc_flag_disables_noisy_metadata_correction(self, params, grads):
        step, state = rmsprop(params, lr=1e-2, noise_bias_correction=False)
        _, state = step(
            noised(grads, max_norm=1.0, noise_stddev=0.5),
            state,
            params=params,
        )
        assert _rms_state(state).phi == 0.0


class TestSecondMomentMode:
    @pytest.fixture
    def sq_grads(self, grads):
        return {k: v.pow(2) + 0.01 for k, v in grads.items()}

    def test_consumes_external_g_squared(self, params, grads, sq_grads):
        alpha = 0.99
        step, state = rmsprop(params, lr=1e-2, alpha=alpha)
        output = SecondMomentNoiseOutput(
            noised(grads, max_norm=1.0, noise_stddev=0.1),
            noised(sq_grads, max_norm=1.0, noise_stddev=0.1),
        )
        _, state = step(output, state, params=params)
        st = _rms_state(state)
        for k in params:
            torch.testing.assert_close(st.nu[k], (1 - alpha) * sq_grads[k])

    def test_negative_squared_stream_bounded(self, params, grads):
        sq = {k: -torch.ones_like(v) for k, v in grads.items()}
        step, state = rmsprop(params, lr=1e-2)
        output = SecondMomentNoiseOutput(
            noised(grads, max_norm=1.0, noise_stddev=0.1),
            noised(sq, max_norm=1.0, noise_stddev=0.1),
        )
        updates, _ = step(output, state, params=params)
        for k in updates:
            assert torch.isfinite(updates[k]).all()
            assert updates[k].abs().max().item() < 10.0

    def test_explicit_second_moment_kwarg_rejected(self, params, grads, sq_grads):
        step, state = rmsprop(params, lr=1e-2)
        with pytest.raises(TypeError, match="noisy_squared_grads"):
            step(
                grads,
                state,
                params=params,
                noisy_squared_grads=sq_grads,
            )


class TestValidation:
    def test_negative_alpha_raises(self):
        with pytest.raises(ValueError, match="invalid RMSprop"):
            rmsprop({"w": torch.ones(1)}, alpha=-0.1)

    def test_alpha_one_raises(self):
        with pytest.raises(ValueError, match="invalid RMSprop"):
            rmsprop({"w": torch.ones(1)}, alpha=1.0)

    def test_negative_eps_raises(self):
        with pytest.raises(ValueError, match="invalid RMSprop"):
            rmsprop({"w": torch.ones(1)}, eps=0.0)
