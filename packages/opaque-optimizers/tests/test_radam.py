"""Tests for opaque.optimizers.radam — Rectified Adam with DP bias correction.

Two execution branches gate the optimizer:

- Early phase (``ρ_t ≤ 5``): SGD-of-momentum.  No second-moment scaling,
  naturally DP-robust.
- Rectified phase (``ρ_t > 5``): Adam-shaped update with the
  rectification factor ``r_t`` and (optionally) φ-EMA bias correction.

Tests assert that the φ-EMA advances *through* the early branch so the
correction at the first rectified step reflects all prior noise
contributions to ``v``.
"""

from __future__ import annotations

import pytest
import torch

from opaque.api.optimizers._radam import _rectification, _rho_t
from opaque.optimizers import apply_updates, radam
from opaque.optimizers.types import RAdamState
from opaque.types import (
    PerGroup,
    SecondMomentNoiseOutput,
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


class TestRectificationMath:
    """``ρ_t`` and ``r_t`` are pure functions of ``(β₂, t)``."""

    def test_rho_inf_matches_paper(self):
        """``ρ_∞ = 2/(1−β₂) − 1``; β₂=0.999 → ρ_∞ = 1999."""
        b2 = 0.999
        rho_at_huge_t = _rho_t(b2, 10_000)
        expected_rho_inf = 2.0 / (1.0 - b2) - 1.0
        assert rho_at_huge_t == pytest.approx(expected_rho_inf, rel=1e-3)

    def test_rectification_undefined_in_warmup(self):
        """``r_t`` is None when ``ρ_t ≤ 5``."""
        for t in [1, 2, 3, 4, 5]:
            assert _rectification(b2=0.999, t=t) is None

    def test_rectification_defined_long_term(self):
        """For ``t`` past warmup, ``r_t`` is real and converges to 1."""
        b2 = 0.999
        r_20 = _rectification(b2, 20)
        assert r_20 is not None
        assert 0 < r_20 < 1
        r_huge = _rectification(b2, 10_000)
        assert r_huge == pytest.approx(1.0, rel=1e-2)


class TestVanillaRAdam:
    def test_state_init(self, params):
        _step, state = radam(params, lr=1e-3)
        assert isinstance(state, RAdamState)
        assert state.step == 0
        assert state.phi == 0.0
        assert set(state.mu) == set(params)
        assert set(state.nu) == set(params)

    def test_step_increments(self, params, grads):
        step, state = radam(params, lr=1e-3)
        for expected in [1, 2, 3]:
            _, state = step(grads, state, params=params)
            assert state.step == expected

    def test_apply_updates_changes_params(self, params, grads):
        # Run past warmup so second-moment scaling is active.
        step, state = radam(params, lr=1e-2)
        p = {k: v.clone() for k, v in params.items()}
        for _ in range(20):
            updates, state = step(grads, state, params=p)
            p = apply_updates(p, updates)
        assert any(not torch.equal(p[k], params[k]) for k in params)

    def test_finite_updates_through_warmup(self, params, grads):
        """Updates remain finite across the warmup→rectified transition."""
        step, state = radam(params, lr=1e-3)
        for _ in range(15):
            updates, state = step(grads, state, params=params)
            for k in params:
                assert torch.isfinite(updates[k]).all()


class TestNoiseBiasCorrection:
    def test_phi_advances_in_warmup(self, params, grads):
        """φ-EMA must advance during the early SGD-of-momentum branch so
        the first rectified step has a fully-warmed correction."""
        b2 = 0.999
        sigma = 0.5
        step, state = radam(params, lr=1e-3, noise_bias_correction=True)
        expected_phi = 0.0
        # First 5 steps are warmup for β₂=0.999.
        for _ in range(5):
            _, state = step(
                noised(grads, max_norm=1.0, noise_stddev=sigma),
                state,
                params=params,
            )
            expected_phi = b2 * expected_phi + (1 - b2) * (sigma**2)
        phi = state.phi
        assert isinstance(phi, dict)
        assert all(v == pytest.approx(expected_phi) for v in phi.values())
        # And we are still in warmup.
        assert _rectification(b2, state.step) is None

    def test_phi_stays_zero_when_bc_off(self, params, grads):
        step, state = radam(params, lr=1e-3, noise_bias_correction=False)
        _, state = step(
            noised(grads, max_norm=1.0, noise_stddev=0.5),
            state,
            params=params,
        )
        assert state.phi == 0.0

    def test_phi_stays_zero_under_clean_grads(self, params, grads):
        step, state = radam(params, lr=1e-3, noise_bias_correction=True)
        for _ in range(5):
            _, state = step(grads, state, params=params)
        phi = state.phi
        if isinstance(phi, dict):
            assert all(v == 0.0 for v in phi.values())
        else:
            assert phi == 0.0

    def test_bc_changes_rectified_update(self, params, grads):
        """Past warmup, BC must change the update relative to no-BC."""
        step_bc, s_bc = radam(params, lr=1e-3, noise_bias_correction=True)
        step_no, s_no = radam(params, lr=1e-3, noise_bias_correction=False)
        # Drive both past warmup under the same noisy stream.
        for _ in range(12):
            g = noised(grads, max_norm=1.0, noise_stddev=0.5)
            _, s_bc = step_bc(g, s_bc, params=params)
            _, s_no = step_no(g, s_no, params=params)
        assert _rectification(0.999, s_bc.step) is not None
        u_bc, _ = step_bc(
            noised(grads, max_norm=1.0, noise_stddev=0.5), s_bc, params=params
        )
        u_no, _ = step_no(
            noised(grads, max_norm=1.0, noise_stddev=0.5), s_no, params=params
        )
        assert any(not torch.allclose(u_bc[k], u_no[k]) for k in params)


class TestPerGroupBC:
    def test_per_group_routes_phi(self, params, grads):
        stddev = PerGroup(
            groups={("weight",): "w", ("bias",): "b"},
            values={"w": 1.0, "b": 0.1},
        )
        step, state = radam(params, lr=1e-3, noise_bias_correction=True)
        for _ in range(5):
            _, state = step(
                noised(grads, max_norm=1.0, noise_stddev=stddev),
                state,
                params=params,
            )
        phi = state.phi
        assert isinstance(phi, dict)
        # Distinct noise levels must yield distinct φ values.
        vals = list(phi.values())
        assert len({round(v, 12) for v in vals}) > 1

    def test_phi_dict_diverges_per_group(self, params, grads):
        # φ is an EMA of σ² per group, so the steady-state ratio between
        # groups must equal the squared stddev ratio — a σ-instead-of-σ²
        # or per-group mis-scaling bug passes a distinctness check but
        # fails this pin.  RAdam inlines its own per-group EMA rather than
        # sharing update_phi_ema, so it needs its own quantitative test.
        stddev = PerGroup(
            groups={("weight",): "attn", ("bias",): "mlp"},
            values={"attn": 0.2, "mlp": 0.8},
        )
        step, state = radam(params, lr=1e-3, noise_bias_correction=True)
        for _ in range(50):
            _, state = step(
                noised(grads, max_norm=1.0, noise_stddev=stddev),
                state,
                params=params,
            )
        phi = state.phi
        assert isinstance(phi, dict)
        # Variance ratio is (0.8/0.2)² = 16.
        assert phi[("weight",)] > 0
        assert phi[("bias",)] / phi[("weight",)] == pytest.approx(16.0, rel=1e-3)


class TestSecondMomentSubstitution:
    def test_substitution_runs_and_freezes_phi(self, params, grads):
        sq = {k: v.pow(2) + 0.05 for k, v in grads.items()}
        step, state = radam(params, lr=1e-3, noise_bias_correction=True)
        output = SecondMomentNoiseOutput(
            noised(grads, max_norm=1.0, noise_stddev=0.1),
            noised(sq, max_norm=1.0, noise_stddev=0.1),
        )
        updates, state = step(output, state, params=params)
        for k in params:
            assert torch.isfinite(updates[k]).all()
        # Private second-moment path does not advance φ.
        phi = state.phi
        if isinstance(phi, dict):
            assert all(v == 0.0 for v in phi.values())
        else:
            assert phi == 0.0


class TestValidation:
    def test_eps_positive(self, params):
        with pytest.raises(ValueError, match="eps"):
            radam(params, eps=0.0)

    def test_betas_two_values(self, params):
        with pytest.raises(ValueError, match="exactly two"):
            radam(params, betas=(0.9,))  # type: ignore[arg-type]

    def test_beta_range(self, params):
        with pytest.raises(ValueError, match="beta_2"):
            radam(params, betas=(0.9, 1.0))

    def test_negative_weight_decay(self, params):
        with pytest.raises(ValueError, match="weight_decay"):
            radam(params, weight_decay=-1.0)

    def test_update_rms_clip_positive(self, params):
        with pytest.raises(ValueError, match="update_rms_clip"):
            radam(params, update_rms_clip=0.0)
