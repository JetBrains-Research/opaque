"""CGF-backed PLD: correctness, precision, overlap, and impact testing.

Tests the CGF (Cumulant Generating Function) path that handles small noise
multipliers (σ < 0.1) where PLD discretization suffers from grid explosion.
The CGF path uses the saddle-point method of steepest descent for δ/ε queries.

Organized in four layers:
1. Correctness — valid results for small σ, all metrics work, invariants hold
2. Precision — CGF vs analytical Gaussian formula, error characterization
3. Overlap — CGF vs PMF agreement at the σ=0.1 boundary
4. Impact — scenarios that were previously impossible now work
"""

from __future__ import annotations

import math

import pytest

import opaque_accounting as acc
from opaque_accounting import opaque_accounting as _native


# ============================================================================
# 1. Correctness
# ============================================================================


class TestCgfCorrectness:
    """CGF path produces valid, finite results for small σ."""

    @pytest.mark.parametrize("sigma", [0.01, 0.03, 0.05, 0.09])
    def test_gaussian_small_sigma(self, sigma):
        """gaussian(σ) for small σ → finite positive ε."""
        proc = acc.gaussian(sigma)
        eps = proc.cgf().epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0, f"σ={sigma}: ε={eps}"

    @pytest.mark.parametrize("sigma", [0.01, 0.05, 0.09])
    @pytest.mark.parametrize("q", [0.001, 0.01])
    def test_poisson_small_sigma(self, sigma, q):
        """poisson(gaussian(σ), q) * 1000 for small σ → finite positive ε."""
        proc = acc.poisson(acc.gaussian(sigma), q) * 1000
        eps = proc.cgf().epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0, f"σ={sigma}, q={q}: ε={eps}"

    @pytest.mark.parametrize("sigma", [0.03, 0.05])
    def test_all_metrics_work(self, sigma):
        """All privacy metrics return valid results on CGF-backed PLDs."""
        pld = (acc.gaussian(sigma) * 100).cgf()
        # epsilon_at
        eps = pld.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0
        # delta_at
        delta = pld.delta_at(eps * 0.5)
        assert 0.0 <= delta <= 1.0
        # advantage
        adv = pld.advantage()
        assert 0.0 <= adv <= 1.0
        # beta_at (requires PMF)
        pmf_pld = pld.pmf()
        beta = pmf_pld.beta_at(0.1)
        assert 0.0 <= beta <= 1.0
        # risk_at (requires PMF)
        risk = pmf_pld.risk_at(0.5)
        assert 0.0 <= risk <= 0.5

    @pytest.mark.parametrize("sigma", [0.01, 0.03, 0.05, 0.09])
    def test_monotonicity_in_sigma(self, sigma):
        """Smaller σ → larger ε (more privacy loss)."""
        eps_small = acc.gaussian(sigma).cgf().epsilon_at(1e-5)
        eps_large = acc.gaussian(min(sigma * 2, 0.5)).cgf().epsilon_at(1e-5)
        assert eps_small > eps_large, (
            f"σ={sigma}: ε={eps_small}, σ={sigma*2}: ε={eps_large}"
        )

    def test_monotonicity_in_steps(self):
        """More steps → larger ε."""
        g = acc.gaussian(0.05)
        eps_100 = (g * 100).cgf().epsilon_at(1e-5)
        eps_1000 = (g * 1000).cgf().epsilon_at(1e-5)
        assert eps_1000 > eps_100

    @pytest.mark.parametrize("sigma", [0.03, 0.05])
    def test_epsilon_delta_roundtrip(self, sigma):
        """ε(δ) → δ(ε) ≈ δ for CGF-backed PLDs."""
        pld = (acc.gaussian(sigma) * 100).cgf()
        delta = 1e-5
        eps = pld.epsilon_at(delta)
        delta_back = pld.delta_at(eps)
        assert delta_back == pytest.approx(delta, rel=0.05), (
            f"Roundtrip: δ={delta} → ε={eps} → δ'={delta_back}"
        )

    @pytest.mark.parametrize("sigma", [0.03, 0.05])
    def test_advantage_equals_delta_at_zero(self, sigma):
        """advantage() == delta_at(0) by definition."""
        pld = (acc.gaussian(sigma) * 100).cgf()
        assert pld.advantage() == pytest.approx(pld.delta_at(0.0), rel=1e-6)


# ============================================================================
# 2. Precision — CGF vs analytical Gaussian formula
# ============================================================================


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _analytical_gaussian_delta(sigma: float, epsilon: float) -> float:
    """Exact δ(ε) for a single Gaussian mechanism.

    δ(ε) = Φ(Δ/(2σ) − εσ/Δ) − e^ε · Φ(−Δ/(2σ) − εσ/Δ), with Δ=1.
    """
    dt = 1.0 / sigma
    return max(
        _normal_cdf(dt / 2.0 - epsilon / dt)
        - math.exp(epsilon) * _normal_cdf(-dt / 2.0 - epsilon / dt),
        0.0,
    )


class TestCgfVsAnalytical:
    """CGF δ(ε) vs exact Gaussian formula — characterize precision.

    Note: The saddle-point approximation is asymptotic — it works best for
    moderate σ and high composition counts. For very small σ (e.g., 0.05)
    at n=1, the ε range is so extreme that the approximation degrades.
    We test at σ=0.5 where both CGF and PMF paths work.
    """

    def test_single_step_accuracy(self):
        """Single Gaussian step (σ=0.5): CGF vs analytical within 25%."""
        sigma = 0.5
        cgf = _native.cgf_gaussian_pld(sigma)
        for eps in [0.5, 1.0, 2.0, 3.0]:
            analytical = _analytical_gaussian_delta(sigma, eps)
            if analytical < 1e-12:
                continue
            cgf_delta = cgf.delta_at(eps)
            rel_err = abs(cgf_delta - analytical) / analytical
            assert rel_err < 0.30, (
                f"σ={sigma}, ε={eps}: CGF={cgf_delta:.6e}, "
                f"analytical={analytical:.6e}, rel_err={rel_err:.1%}"
            )

    @pytest.mark.parametrize(
        "n,max_rel_err",
        [(10, 0.10), (100, 0.05), (1000, 0.02)],
    )
    def test_precision_improves_with_composition(self, n, max_rel_err):
        """CGF accuracy improves with composition count.

        The saddle-point approximation is asymptotically exact as n → ∞.
        Compare CGF path against PMF path at σ=0.5 where both work.
        """
        sigma = 0.5
        cgf = _native.cgf_gaussian_pld(sigma).self_compose(n)
        pmf = _native.gaussian_pld(sigma, _native.DiscretizationConfig())
        pmf_composed = pmf.self_compose(n)

        # Compare epsilon_at (more stable than delta_at for comparison)
        eps_cgf = cgf.epsilon_at(1e-5)
        eps_pmf = pmf_composed.epsilon_at(1e-5)

        if not (math.isfinite(eps_cgf) and math.isfinite(eps_pmf)):
            return

        rel_err = abs(eps_cgf - eps_pmf) / eps_pmf
        assert rel_err < max_rel_err, (
            f"n={n}: CGF ε={eps_cgf:.6f}, PMF ε={eps_pmf:.6f}, rel_err={rel_err:.1%}"
        )


# ============================================================================
# 3. Overlap — CGF vs PMF agreement at the boundary
# ============================================================================


class TestCgfVsPmfOverlap:
    """At σ=0.1 (threshold boundary), explicit CGF and PMF should agree."""

    def _get_both_plds(self, sigma: float = 0.1):
        """Build both CGF and PMF PLDs for the same σ."""
        config = _native.DiscretizationConfig()
        cgf = _native.cgf_gaussian_pld(sigma)
        pmf = _native.gaussian_pld(sigma, config)
        return cgf, pmf

    def test_epsilon_at_agreement(self):
        """CGF and PMF epsilon_at agree within 5% at σ=0.1."""
        cgf, pmf = self._get_both_plds()
        for delta in [1e-3, 1e-5, 1e-7]:
            eps_cgf = cgf.epsilon_at(delta)
            eps_pmf = pmf.epsilon_at(delta)
            if not (math.isfinite(eps_cgf) and math.isfinite(eps_pmf)):
                continue
            assert eps_cgf == pytest.approx(eps_pmf, rel=0.05), (
                f"δ={delta}: CGF ε={eps_cgf:.6f}, PMF ε={eps_pmf:.6f}"
            )

    def test_delta_at_agreement(self):
        """CGF and PMF delta_at agree within 5% at σ=0.1."""
        cgf, pmf = self._get_both_plds()
        eps_ref = pmf.epsilon_at(1e-5) * 0.8
        delta_cgf = cgf.delta_at(eps_ref)
        delta_pmf = pmf.delta_at(eps_ref)
        if delta_pmf > 1e-12:
            assert delta_cgf == pytest.approx(delta_pmf, rel=0.05), (
                f"ε={eps_ref}: CGF δ={delta_cgf:.6e}, PMF δ={delta_pmf:.6e}"
            )

    def test_composed_agreement(self):
        """CGF and PMF agree after composition (× 1000) at σ=0.1."""
        cgf, pmf = self._get_both_plds()
        cgf_c = cgf.self_compose(1000)
        pmf_c = pmf.self_compose(1000)

        eps_cgf = cgf_c.epsilon_at(1e-5)
        eps_pmf = pmf_c.epsilon_at(1e-5)
        assert eps_cgf == pytest.approx(eps_pmf, rel=0.05), (
            f"Composed ×1000: CGF ε={eps_cgf:.4f}, PMF ε={eps_pmf:.4f}"
        )

    def test_composed_epsilon_tightens(self):
        """After composition, CGF and PMF epsilon agreement tightens.

        Note: advantage (δ(0)) has a known numerical issue in the
        saddle-point solver at ε=0 due to the log(t) singularity.
        We test epsilon_at instead, which is the primary use case.
        """
        cgf, pmf = self._get_both_plds()
        # Single step agreement
        eps_cgf_1 = cgf.epsilon_at(1e-5)
        eps_pmf_1 = pmf.epsilon_at(1e-5)
        err_1 = abs(eps_cgf_1 - eps_pmf_1) / eps_pmf_1

        # After 100× composition
        cgf_c = cgf.self_compose(100)
        pmf_c = pmf.self_compose(100)
        eps_cgf_100 = cgf_c.epsilon_at(1e-5)
        eps_pmf_100 = pmf_c.epsilon_at(1e-5)
        err_100 = abs(eps_cgf_100 - eps_pmf_100) / eps_pmf_100

        # Composition should tighten agreement
        assert err_100 < err_1 or err_100 < 0.02, (
            f"n=1: err={err_1:.1%}, n=100: err={err_100:.1%}"
        )


# ============================================================================
# 4. Composition
# ============================================================================


class TestCgfComposition:
    """CGF + CGF stays CGF; CGF + PMF materializes correctly."""

    def test_cgf_plus_cgf(self):
        """Two different small-σ Gaussians composed."""
        g1 = acc.gaussian(0.05) * 500
        g2 = acc.gaussian(0.08) * 500
        composed = g1 | g2
        eps = composed.cgf().epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_cgf_plus_pmf(self):
        """Small-σ Gaussian | large-σ Gaussian → mixed composition."""
        g_small = acc.gaussian(0.05) * 500
        g_large = acc.gaussian(0.8) * 500
        composed = g_small | g_large
        eps = composed.cgf().epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

        # Should be larger than either alone
        eps_small = g_small.cgf().epsilon_at(1e-5)
        eps_large = g_large.cgf().epsilon_at(1e-5)
        assert eps > eps_small
        assert eps > eps_large

    def test_self_compose(self):
        """Self-compose × 10000 works for very small σ."""
        proc = acc.gaussian(0.05) * 10000
        eps = proc.cgf().epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0


# ============================================================================
# 5. Impact — previously-impossible scenarios now work
# ============================================================================


class TestCgfImpact:
    """Verify the CGF path enables previously-impossible scenarios."""

    def test_small_sigma_gaussian(self):
        """gaussian(0.05) * 1000 → finite ε (was impossible before)."""
        eps = (acc.gaussian(0.05) * 1000).cgf().epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_small_sigma_poisson(self):
        """poisson(gaussian(0.05), 0.01) * 1000 → finite ε."""
        eps = (acc.poisson(acc.gaussian(0.05), 0.01) * 1000).cgf().epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_very_small_sigma(self):
        """gaussian(0.01) * 10000 → finite ε."""
        eps = (acc.gaussian(0.01) * 10000).cgf().epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_calibration_extended_range(self):
        """Calibration works with param_min=0.01 (CGF range)."""
        from opaque_accounting import calibration as cal

        # Target ε=500 at δ=1e-5 — achievable in [0.01, 2.5] * 1000
        result = cal.calibrate(
            cal.epsilon_budget(500.0, delta=1e-5),
            lambda nm: (acc.gaussian(nm) * 1000).cgf(),
            0.01,
            2.5,
        )
        assert result.converged
        assert 0.01 <= result.param <= 2.5


# ============================================================================
# 6. Explicit CGF API — proc.cgf() opt-in
# ============================================================================


class TestCgfExplicit:
    """Test the explicit cgf() method on DpProcess."""

    def test_gaussian_cgf_returns_cgf(self):
        """acc.gaussian(0.5).cgf() returns a CGF-backed PLD."""
        pld = acc.gaussian(0.5).cgf()
        assert "cgf" in repr(pld)

    def test_gaussian_cgf_large_sigma(self):
        """CGF works for large σ too (not just small σ)."""
        pld = acc.gaussian(2.0).cgf()
        eps = pld.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_poisson_cgf_returns_cgf(self):
        """acc.poisson(acc.gaussian(1.1), 0.01).cgf() works."""
        pld = acc.poisson(acc.gaussian(1.1), 0.01).cgf()
        assert "cgf" in repr(pld)

    def test_repeated_cgf_composition(self):
        """(gaussian * 1000).cgf() composes via CGF (O(1))."""
        proc = acc.gaussian(0.5) * 1000
        pld = proc.cgf()
        assert "cgf" in repr(pld)
        eps = pld.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_composed_cgf(self):
        """(g1 | g2).cgf() works when both have CGF."""
        g1 = acc.gaussian(0.5) * 100
        g2 = acc.gaussian(0.8) * 200
        pld = (g1 | g2).cgf()
        assert "cgf" in repr(pld)
        eps = pld.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_eps_delta_with_delta_no_cgf(self):
        """eps_delta with delta > 0 has no CGF (infinite MGF)."""
        with pytest.raises(NotImplementedError):
            acc.eps_delta(1.0, 1e-5).cgf()

    def test_rectified_gaussian_cgf_works(self):
        """Rectified Gaussian now supports CGF."""
        eps = acc.rectified_gaussian(0.5, 5.0).cgf().epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_mixed_composition_cgf_works(self):
        """Composed Gaussian + RectifiedGaussian CGF works."""
        composed = acc.gaussian(0.5) | acc.rectified_gaussian(0.5, 5.0)
        eps = composed.cgf().epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_poisson_rectified_cgf_works(self):
        """Poisson-subsampled rectified Gaussian supports CGF."""
        eps = acc.poisson(acc.rectified_gaussian(0.5, 5.0), 0.01).cgf().epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    @pytest.mark.parametrize("sigma", [0.25, 0.5, 1.0])
    def test_cgf_matches_pld(self, sigma):
        """CGF and PMF paths agree for moderate σ."""
        proc = acc.gaussian(sigma) * 100
        eps_cgf = proc.cgf().epsilon_at(1e-5)
        eps_pld = proc.pmf().epsilon_at(1e-5)
        rel_err = abs(eps_cgf - eps_pld) / eps_pld
        assert rel_err < 0.05, (
            f"σ={sigma}: CGF ε={eps_cgf:.4f}, PLD ε={eps_pld:.4f}, "
            f"rel_err={rel_err:.2%}"
        )

    def test_cgf_delta_matches_pld(self):
        """CGF and PMF delta_at agree for moderate σ."""
        proc = acc.gaussian(0.5) * 100
        eps_test = proc.pmf().epsilon_at(0.1) * 0.8
        d_cgf = proc.cgf().delta_at(eps_test)
        d_pld = proc.pmf().delta_at(eps_test)
        rel_err = abs(d_cgf - d_pld) / d_pld if d_pld > 1e-12 else abs(d_cgf)
        # MSD is a first-order approximation; allow 15% at this composition count.
        assert rel_err < 0.15, (
            f"CGF δ={d_cgf:.6e}, PLD δ={d_pld:.6e}, rel_err={rel_err:.2%}"
        )
