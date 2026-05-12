"""Tests for Xiang et al. (2025) order-statistics privacy auditing.

Covers ``_gaussian_trade_off.py``, ``_xiang.py``, and the modified
``epsilon_at()`` / ``epsilon_at_gaussian()`` methods on ``OneRunEstimate``.
"""

from __future__ import annotations

import ast
import math
import pathlib

import numpy as np
import pytest
import scipy.special
import scipy.stats

from opaque.api.auditing._gaussian_trade_off import (
    gaussian_base_pair_grid,
    gaussian_to_eps_delta,
)
from opaque.api.auditing._xiang import (
    xiang_p_value_eps_delta,
    xiang_p_value_gaussian,
)
from opaque.auditing import one_run
from opaque.auditing.types import CoinFlip


# ---- Helpers ---------------------------------------------------------------

def _make_estimate(in_scores, out_scores):
    """Build a OneRunEstimate from raw in/out score arrays."""
    in_scores = np.asarray(in_scores, dtype=float)
    out_scores = np.asarray(out_scores, dtype=float)
    n_in, n_out = len(in_scores), len(out_scores)
    canary_indices = np.arange(n_in + n_out)
    mask = np.array([True] * n_in + [False] * n_out)
    cf = CoinFlip(
        num_canaries=n_in + n_out,
        canary_indices=canary_indices,
        _in_mask=mask,
        in_indices=canary_indices[mask],
        out_indices=canary_indices[~mask],
    )
    scores = np.empty(n_in + n_out)
    scores[mask] = in_scores
    scores[~mask] = out_scores
    return one_run(scores, coin_flip=cf)


def _delta_at(mu: float, eps: float) -> float:
    """Analytical δ(ε) for μ-GDP."""
    a = mu / 2.0 - eps / mu
    b = -mu / 2.0 - eps / mu
    return scipy.stats.norm.cdf(a) - math.exp(eps) * scipy.stats.norm.cdf(b)


# ---- gaussian_to_eps_delta -------------------------------------------------

class TestGaussianToEpsDelta:
    """Unit tests for GDP-to-(ε,δ) conversion."""

    def test_mu_zero_returns_zero(self):
        assert gaussian_to_eps_delta(0.0, 1e-5) == 0.0

    @pytest.mark.parametrize("mu", [0.5, 1.0, 2.0, 5.0])
    def test_cross_check(self, mu):
        """Returned ε satisfies δ(ε) ≈ target δ."""
        delta = 1e-5
        eps = gaussian_to_eps_delta(mu, delta)
        actual_delta = _delta_at(mu, eps)
        assert abs(actual_delta - delta) < 1e-7, (
            f"mu={mu}: delta mismatch {actual_delta} vs {delta}"
        )

    def test_monotone_in_mu(self):
        """Larger μ → larger ε at fixed δ."""
        delta = 1e-5
        prev = 0.0
        for mu in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            eps = gaussian_to_eps_delta(mu, delta)
            assert eps >= prev, f"not monotone at mu={mu}"
            prev = eps

    def test_large_delta(self):
        """Large δ → small ε (less privacy needed)."""
        eps = gaussian_to_eps_delta(1.0, 0.5)
        assert eps < gaussian_to_eps_delta(1.0, 1e-5)

    def test_negative_mu_raises(self):
        with pytest.raises(ValueError, match="mu must be >= 0"):
            gaussian_to_eps_delta(-1.0, 1e-5)

    def test_delta_out_of_range_raises(self):
        with pytest.raises(ValueError, match="delta must be in"):
            gaussian_to_eps_delta(1.0, 0.0)
        with pytest.raises(ValueError, match="delta must be in"):
            gaussian_to_eps_delta(1.0, 1.0)
        with pytest.raises(ValueError, match="delta must be in"):
            gaussian_to_eps_delta(1.0, -0.1)


# ---- gaussian_base_pair_grid -----------------------------------------------

class TestGaussianBasePairGrid:
    """Unit tests for the discretised Gaussian base pair."""

    @pytest.mark.parametrize("mu", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_total_mass_near_one(self, mu):
        grid = gaussian_base_pair_grid(mu, 10_000)
        total = float(np.sum(grid.mass))
        assert abs(total - 1.0) < 0.01, f"mu={mu}: mass={total}"

    @pytest.mark.parametrize("mu", [0.5, 1.0, 2.0])
    def test_F_y_monotone(self, mu):
        grid = gaussian_base_pair_grid(mu, 10_000)
        assert np.all(np.diff(grid.F_y) >= -1e-10)

    @pytest.mark.parametrize("mu", [0.5, 1.0, 2.0])
    def test_abs_pl_sorted(self, mu):
        grid = gaussian_base_pair_grid(mu, 10_000)
        assert np.all(np.diff(grid.abs_privacy_loss) >= -1e-10)

    def test_min_abs_pl_near_zero(self):
        grid = gaussian_base_pair_grid(1.0, 10_000)
        assert grid.abs_privacy_loss[0] < 0.05

    def test_mass_non_negative(self):
        grid = gaussian_base_pair_grid(1.0, 10_000)
        assert np.all(grid.mass >= -1e-15)

    def test_mu_zero_raises(self):
        with pytest.raises(ValueError, match="mu must be > 0"):
            gaussian_base_pair_grid(0.0, 100)


# ---- xiang_p_value_eps_delta -----------------------------------------------

class TestXiangPValueEpsDelta:
    """Unit tests for the (ε,δ)-DP analytical p-value."""

    def test_matches_binom_cdf_delta_zero(self):
        """At δ = 0, should equal binom.cdf exactly."""
        r, u, eps = 1000, 400, 1.0
        p = scipy.special.expit(-eps)
        expected = scipy.stats.binom.cdf(u, r, p)
        actual = xiang_p_value_eps_delta(r, u, eps, 0.0)
        assert abs(actual - expected) < 1e-10

    def test_matches_binom_cdf_delta_positive(self):
        """At δ > 0, should equal binom.cdf with reduced n_eff."""
        r, u, eps, delta = 1000, 400, 1.0, 0.01
        p = scipy.special.expit(-eps)
        n_eff = r - round(r * delta)
        expected = scipy.stats.binom.cdf(u, n_eff, p)
        actual = xiang_p_value_eps_delta(r, u, eps, delta)
        assert abs(actual - expected) < 1e-10

    def test_monotone_in_eps(self):
        """Larger ε → larger p-value (less surprising errors)."""
        p1 = xiang_p_value_eps_delta(1000, 300, 0.5, 0.0)
        p2 = xiang_p_value_eps_delta(1000, 300, 1.0, 0.0)
        p3 = xiang_p_value_eps_delta(1000, 300, 2.0, 0.0)
        assert p1 <= p2 <= p3

    def test_perfect_attack(self):
        """u = 0 → very small p-value."""
        p = xiang_p_value_eps_delta(1000, 0, 0.5, 0.0)
        assert p < 1e-10

    def test_random_guessing(self):
        """u = r/2 at ε = 0 → p-value ≈ 0.5."""
        p = xiang_p_value_eps_delta(1000, 500, 0.0, 0.0)
        assert 0.4 < p < 0.6

    def test_delta_one_returns_one(self):
        """δ = 1 → n_eff = 0 → p = 1."""
        assert xiang_p_value_eps_delta(100, 10, 1.0, 1.0) == 1.0

    def test_worse_than_expected(self):
        """u ≫ expected → p-value > 0.5."""
        p = xiang_p_value_eps_delta(100, 80, 1.0, 0.0)
        assert p > 0.5


# ---- xiang_p_value_gaussian ------------------------------------------------

class TestXiangPValueGaussian:
    """Unit tests for the Gaussian grid-based p-value."""

    def test_perfect_attack(self):
        p = xiang_p_value_gaussian(100, 100, 0, 1.0, 5000)
        assert 0 < p < 0.01

    def test_near_random(self):
        p = xiang_p_value_gaussian(100, 100, 45, 1.0, 5000)
        assert p > 0.3

    def test_mu_zero_returns_one(self):
        assert xiang_p_value_gaussian(100, 100, 10, 0.0) == 1.0

    def test_monotone_in_mu(self):
        """Larger μ → less private → more expected correct → larger p-value."""
        prev = 0.0
        for mu in [0.3, 0.5, 1.0, 2.0, 3.0]:
            p = xiang_p_value_gaussian(100, 100, 20, mu, 5000)
            assert p >= prev - 1e-10, f"not monotone at mu={mu}"
            prev = p

    def test_tighter_than_eps_delta(self):
        """At the same effective (ε,δ), Gaussian p-value ≤ (ε,δ) p-value."""
        mu = 1.0
        delta = 1e-5
        eps = gaussian_to_eps_delta(mu, delta)
        p_gauss = xiang_p_value_gaussian(200, 200, 30, mu, 5000)
        p_epsd = xiang_p_value_eps_delta(200, 30, eps, delta)
        assert p_gauss <= p_epsd + 1e-10

    def test_scaling_with_n(self):
        """More canaries → smaller p-value (more evidence)."""
        p50 = xiang_p_value_gaussian(50, 50, 0, 1.0, 5000)
        p200 = xiang_p_value_gaussian(200, 200, 0, 1.0, 5000)
        assert p200 < p50

    def test_returns_float(self):
        p = xiang_p_value_gaussian(100, 100, 20, 1.0, 5000)
        assert isinstance(p, float)
        assert 0 <= p <= 1


# ---- epsilon_at (Xiang dispatch) -------------------------------------------

class TestEpsilonAtXiang:
    """Behavioural tests for the replaced epsilon_at()."""

    def test_separated_delta_zero(self):
        est = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        eps = est.epsilon_at(delta=0.0)
        assert eps > 0

    def test_separated_delta_positive(self):
        est = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        eps = est.epsilon_at(delta=1e-5)
        assert eps > 0

    def test_no_separation(self):
        est = _make_estimate(np.arange(100), np.arange(100))
        eps = est.epsilon_at(delta=0.0)
        assert eps < 1.0

    def test_threshold_works(self):
        est = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        eps = est.epsilon_at(delta=0.0, threshold=50.0)
        assert eps > 0

    def test_dominance_over_steinke(self):
        """Xiang should give ε ≥ Steinke on well-separated data."""
        est = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        eps_xiang = est.epsilon_at(delta=1e-5)
        eps_steinke = est._epsilon_at_steinke(delta=1e-5)
        assert eps_xiang >= eps_steinke * 0.9, (
            f"Xiang ({eps_xiang:.4f}) too much below Steinke ({eps_steinke:.4f})"
        )

    def test_returns_float(self):
        est = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        assert isinstance(est.epsilon_at(delta=0.0), float)
        assert isinstance(est.epsilon_at(delta=0.1), float)

    def test_invalid_significance(self):
        est = _make_estimate([1, 2], [3, 4])
        with pytest.raises(ValueError, match="significance"):
            est.epsilon_at(significance=0.0)

    def test_invalid_delta(self):
        est = _make_estimate([1, 2], [3, 4])
        with pytest.raises(ValueError, match="delta"):
            est.epsilon_at(delta=-0.1)

    def test_large_eps(self):
        """1000 canaries with perfect separation → large ε."""
        est = _make_estimate(np.arange(1000, 2000), np.arange(0, 1000))
        eps = est.epsilon_at(delta=0.0, threshold=1000)
        assert eps > 5.0


# ---- epsilon_at_gaussian ---------------------------------------------------

class TestEpsilonAtGaussian:
    """Tests for the new epsilon_at_gaussian() method."""

    def test_basic(self):
        est = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        eps = est.epsilon_at_gaussian(delta=1e-5)
        assert eps > 0

    def test_tighter_than_eps_delta(self):
        """Gaussian should give ε ≥ (ε,δ)-DP on the same data."""
        est = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        eg = est.epsilon_at_gaussian(delta=1e-5)
        ed = est.epsilon_at(delta=1e-5)
        assert eg >= ed * 0.9, f"gaussian ({eg:.4f}) << eps_delta ({ed:.4f})"

    def test_delta_zero_raises(self):
        est = _make_estimate([1, 2], [3, 4])
        with pytest.raises(ValueError, match="delta > 0"):
            est.epsilon_at_gaussian(delta=0.0)

    def test_delta_negative_raises(self):
        est = _make_estimate([1, 2], [3, 4])
        with pytest.raises(ValueError, match="delta > 0"):
            est.epsilon_at_gaussian(delta=-1.0)

    def test_delta_gt_one_raises(self):
        est = _make_estimate([1, 2], [3, 4])
        with pytest.raises(ValueError, match="delta must be in"):
            est.epsilon_at_gaussian(delta=2.0)

    def test_threshold_works(self):
        est = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        eps = est.epsilon_at_gaussian(delta=1e-5, threshold=50.0)
        assert eps > 0

    def test_returns_float(self):
        est = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        assert isinstance(est.epsilon_at_gaussian(delta=1e-5), float)


# ---- Contract: torch-free --------------------------------------------------

class TestTorchFree:
    """New auditing modules must not import torch."""

    _AUDITING_SRC = (
        pathlib.Path(__file__).resolve().parents[2]
        / "src"
        / "opaque"
        / "api"
        / "auditing"
    )

    @pytest.mark.parametrize(
        "filename",
        ["_gaussian_trade_off.py", "_xiang.py"],
    )
    def test_no_torch_import(self, filename):
        path = self._AUDITING_SRC / filename
        if not path.exists():
            pytest.skip(f"{filename} not found")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("torch"), (
                        f"{filename} imports {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("torch"), (
                    f"{filename} imports from {node.module}"
                )
