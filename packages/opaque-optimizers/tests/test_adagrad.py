"""Tests for opaque.optimizers._adagrad with DP-aware Φ subtraction."""

from __future__ import annotations

import pytest
import torch

from opaque.optimizers import adagrad
from opaque.types import noised


@pytest.fixture
def params():
    torch.manual_seed(0)
    return {"weight": torch.randn(4, 3), "bias": torch.randn(3)}


@pytest.fixture
def grads(params):
    torch.manual_seed(1)
    return {k: torch.randn_like(v) for k, v in params.items()}


class TestVanilla:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"lr": 1e-2, "eps": 1e-10, "initial_accumulator_value": 0.0},
            {"lr": 5e-3, "eps": 1e-8, "initial_accumulator_value": 0.1},
            {"lr": 0.1, "eps": 1e-6, "initial_accumulator_value": 1.0},
        ],
        ids=["default", "warm_accumulator", "large_accumulator"],
    )
    def test_steps_produce_finite_updates(self, params, kwargs):
        step, state = adagrad(params, **kwargs)
        torch.manual_seed(42)
        for _ in range(10):
            step_grads = {k: torch.randn_like(v) for k, v in params.items()}
            updates, state = step(step_grads, state, params=params)
            for k in params:
                assert torch.isfinite(updates[k]).all()

    def test_v_acc_starts_at_initial_value(self, params):
        step, state = adagrad(params, lr=1e-2, initial_accumulator_value=0.5)
        st = state
        for k in params:
            torch.testing.assert_close(st.v_acc[k], torch.full_like(params[k], 0.5))

    def test_v_acc_accumulates_without_decay(self, params, grads):
        step, state = adagrad(params, lr=1e-2, initial_accumulator_value=0.0)
        for _ in range(3):
            _, state = step(grads, state, params=params)
        st = state
        # After 3 identical-grad steps: v_acc = 3 * g²
        for k in grads:
            torch.testing.assert_close(st.v_acc[k], 3 * grads[k] * grads[k])

    def test_phi_acc_zero_without_noise(self, params, grads):
        step, state = adagrad(params, lr=1e-2)
        for _ in range(5):
            _, state = step(grads, state, params=params)
        assert state.phi_acc == 0.0


class TestDPCorrection:
    def test_phi_acc_accumulates_cumulatively(self, params, grads):
        sigma = 0.5
        step, state = adagrad(params, lr=1e-2, noise_bias_correction=True)
        for t in range(1, 6):
            _, state = step(
                noised(grads, max_norm=1.0, noise_stddev=sigma),
                state,
                params=params,
            )
            # Cumulative — every step adds σ², no decay.
            expected = t * (sigma**2)
            phi_acc = state.phi_acc
            assert isinstance(phi_acc, dict)
            assert all(v == pytest.approx(expected) for v in phi_acc.values())

    def test_noisy_updates_take_per_step_metadata(self, params, grads):
        step, state = adagrad(params, lr=1e-2, noise_bias_correction=True)
        expected = 0.0
        for sigma in [0.1, 0.2, 0.3]:
            _, state = step(
                noised(grads, max_norm=1.0, noise_stddev=sigma),
                state,
                params=params,
            )
            expected += sigma**2
        phi_acc = state.phi_acc
        assert isinstance(phi_acc, dict)
        assert all(v == pytest.approx(expected) for v in phi_acc.values())

    def test_correction_prevents_runaway_denominator(self, params):
        """The headline DP-Adagrad fix: with correction, the effective
        v̂ tracks signal contribution only.  Without correction, v_acc
        would carry t·σ² forever.

        We feed zero gradients with ``NoisedPytree`` σ metadata,
        and verify v̂_corrected stays at the floor (no signal → no
        denominator inflation)."""
        zero_grads = {k: torch.zeros_like(v) for k, v in params.items()}
        sigma = 1.0
        step, state = adagrad(params, lr=1e-2, noise_bias_correction=True)
        # Update receives zero gradients plus σ metadata. v_acc grows
        # to 0 (g²=0); φ_acc
        # grows by σ² per step.  v_acc - φ_acc < 0 → clamped to floor.
        for _ in range(20):
            updates, state = step(
                noised(zero_grads, max_norm=1.0, noise_stddev=sigma),
                state,
                params=params,
            )
            # Updates should be ~zero (g/sqrt(floor) ≈ 0 since g=0).
            for k in updates:
                assert torch.all(updates[k].abs() < 1e-3)

    def test_floor_keeps_finite(self):
        params = {"w": torch.ones(3)}
        grads = {"w": torch.ones(3) * 0.01}
        # Huge sigma — phi_acc dominates, denom would be negative
        # without floor.
        step, state = adagrad(params, lr=1e-3, noise_bias_correction=True)
        for _ in range(10):
            updates, state = step(
                noised(grads, max_norm=1.0, noise_stddev=1e3),
                state,
                params=params,
            )
            assert torch.isfinite(updates["w"]).all()

    def test_bc_flag_disables_noisy_metadata_correction(self, params, grads):
        step, state = adagrad(params, lr=1e-2, noise_bias_correction=False)
        _, state = step(
            noised(grads, max_norm=1.0, noise_stddev=0.5),
            state,
            params=params,
        )
        assert state.phi_acc == 0.0


class TestWeightDecay:
    def test_decoupled_with_zero_grad(self):
        params = {"w": torch.ones(4) * 2.0}
        grads = {"w": torch.zeros(4)}
        step, state = adagrad(params, lr=0.1, weight_decay=0.1, initial_accumulator_value=1.0)
        updates, _ = step(grads, state, params=params)
        # g = 0 → moment-scaled = 0; only WD survives.
        # update = -lr * (0 + wd * params) = -0.01 * 2.0
        expected = -0.1 * 0.1 * params["w"]
        torch.testing.assert_close(updates["w"], expected)


class TestValidation:
    def test_negative_eps_raises(self):
        with pytest.raises(ValueError):
            adagrad({"w": torch.ones(1)}, eps=0.0)

    def test_negative_initial_accumulator_raises(self):
        with pytest.raises(ValueError):
            adagrad({"w": torch.ones(1)}, initial_accumulator_value=-1.0)

    def test_negative_weight_decay_raises(self):
        with pytest.raises(ValueError):
            adagrad({"w": torch.ones(1)}, weight_decay=-0.1)
