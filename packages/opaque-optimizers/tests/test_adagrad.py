"""Tests for opaque.optimizers._adagrad with DP-aware Φ subtraction."""

from __future__ import annotations

import pytest
import torch

torchopt = pytest.importorskip("torchopt")

from opaque.optimizers import adagrad
from opaque.optimizers.types import AdagradState
from opaque.types import noised


@pytest.fixture
def params():
    torch.manual_seed(0)
    return {"weight": torch.randn(4, 3), "bias": torch.randn(3)}


@pytest.fixture
def grads(params):
    torch.manual_seed(1)
    return {k: torch.randn_like(v) for k, v in params.items()}


def _ada_state(chain_state) -> AdagradState:
    for entry in chain_state:
        if isinstance(entry, AdagradState):
            return entry
    raise AssertionError(f"AdagradState not found in {chain_state!r}")


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
    def test_matches_torchopt_adagrad(self, params, kwargs):
        """Vanilla Adagrad is numerically identical to torchopt.adagrad."""
        opt_opaque = adagrad(**kwargs)
        opt_ref = torchopt.adagrad(**kwargs)
        state_opaque = opt_opaque.init(params)
        state_ref = opt_ref.init(params)

        torch.manual_seed(42)
        for _ in range(10):
            step_grads = {k: torch.randn_like(v) for k, v in params.items()}
            updates_opaque, state_opaque = opt_opaque.update(
                step_grads, state_opaque, params=params
            )
            updates_ref, state_ref = opt_ref.update(
                step_grads, state_ref, params=params
            )
            for k in params:
                torch.testing.assert_close(updates_opaque[k], updates_ref[k])

    def test_v_acc_starts_at_initial_value(self, params):
        opt = adagrad(lr=1e-2, initial_accumulator_value=0.5)
        st = _ada_state(opt.init(params))
        for k in params:
            torch.testing.assert_close(st.v_acc[k], torch.full_like(params[k], 0.5))

    def test_v_acc_accumulates_without_decay(self, params, grads):
        opt = adagrad(lr=1e-2, initial_accumulator_value=0.0)
        state = opt.init(params)
        for _ in range(3):
            _, state = opt.update(grads, state, params=params)
        st = _ada_state(state)
        # After 3 identical-grad steps: v_acc = 3 * g²
        for k in grads:
            torch.testing.assert_close(st.v_acc[k], 3 * grads[k] * grads[k])

    def test_phi_acc_zero_without_noise(self, params, grads):
        opt = adagrad(lr=1e-2)
        state = opt.init(params)
        for _ in range(5):
            _, state = opt.update(grads, state, params=params)
        assert _ada_state(state).phi_acc == 0.0


class TestDPCorrection:
    def test_phi_acc_accumulates_cumulatively(self, params, grads):
        sigma = 0.5
        opt = adagrad(lr=1e-2, noise_bias_correction=True)
        state = opt.init(params)
        for t in range(1, 6):
            _, state = opt.update(
                noised(grads, max_norm=1.0, noise_stddev=sigma),
                state,
                params=params,
            )
            # Cumulative — every step adds σ², no decay.
            expected = t * (sigma**2)
            assert _ada_state(state).phi_acc == pytest.approx(expected)

    def test_noisy_updates_take_per_step_metadata(self, params, grads):
        opt = adagrad(lr=1e-2, noise_bias_correction=True)
        state = opt.init(params)
        expected = 0.0
        for sigma in [0.1, 0.2, 0.3]:
            _, state = opt.update(
                noised(grads, max_norm=1.0, noise_stddev=sigma),
                state,
                params=params,
            )
            expected += sigma**2
        assert _ada_state(state).phi_acc == pytest.approx(expected)

    def test_correction_prevents_runaway_denominator(self, params):
        """The headline DP-Adagrad fix: with correction, the effective
        v̂ tracks signal contribution only.  Without correction, v_acc
        would carry t·σ² forever.

        We feed zero gradients with ``NoisedPytree`` σ metadata,
        and verify v̂_corrected stays at the floor (no signal → no
        denominator inflation)."""
        zero_grads = {k: torch.zeros_like(v) for k, v in params.items()}
        sigma = 1.0
        opt = adagrad(lr=1e-2, noise_bias_correction=True)
        state = opt.init(params)
        # Update receives zero gradients plus σ metadata. v_acc grows
        # to 0 (g²=0); φ_acc
        # grows by σ² per step.  v_acc - φ_acc < 0 → clamped to floor.
        for _ in range(20):
            updates, state = opt.update(
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
        opt = adagrad(lr=1e-3, noise_bias_correction=True)
        state = opt.init(params)
        for _ in range(10):
            updates, state = opt.update(
                noised(grads, max_norm=1.0, noise_stddev=1e3),
                state,
                params=params,
            )
            assert torch.isfinite(updates["w"]).all()

    def test_bc_flag_disables_noisy_metadata_correction(self, params, grads):
        opt = adagrad(lr=1e-2, noise_bias_correction=False)
        state = opt.init(params)
        _, state = opt.update(
            noised(grads, max_norm=1.0, noise_stddev=0.5),
            state,
            params=params,
        )
        assert _ada_state(state).phi_acc == 0.0


class TestWeightDecay:
    def test_decoupled_with_zero_grad(self):
        params = {"w": torch.ones(4) * 2.0}
        grads = {"w": torch.zeros(4)}
        opt = adagrad(lr=0.1, weight_decay=0.1, initial_accumulator_value=1.0)
        state = opt.init(params)
        updates, _ = opt.update(grads, state, params=params)
        # g = 0 → moment-scaled = 0; only WD survives.
        # update = -lr * (0 + wd * params) = -0.01 * 2.0
        expected = -0.1 * 0.1 * params["w"]
        torch.testing.assert_close(updates["w"], expected)


class TestValidation:
    def test_negative_eps_raises(self):
        with pytest.raises(ValueError, match="eps"):
            adagrad(eps=0.0)

    def test_negative_initial_accumulator_raises(self):
        with pytest.raises(ValueError, match="initial_accumulator_value"):
            adagrad(initial_accumulator_value=-1.0)

    def test_negative_weight_decay_raises(self):
        with pytest.raises(ValueError, match="weight_decay"):
            adagrad(weight_decay=-0.1)
