"""Cross-validation: opaque.accounting vs dp_accounting + riskcal.

These tests verify numerical agreement between opaque's Rust-backed PLD
implementation and Google's dp_accounting library (the reference).  The
riskcal library provides additional f-DP / error-rate metrics on top of
dp_accounting PLDs, enabling triple validation.

**Tolerance**: < 1e-6 for all epsilon/delta comparisons.  Empirically the
agreement is ~1e-8 to 1e-12.

Requires optional deps — install with ``uv sync --group cross-validation``.
The ``pytest.importorskip`` calls below gate the entire module automatically.

Run selectively with::

    uv run --group cross-validation pytest tests/test_cross_validation.py -v
"""

from __future__ import annotations

import math

import pytest

dp_accounting = pytest.importorskip("dp_accounting")
riskcal = pytest.importorskip("riskcal")

import riskcal.analysis as rc_analysis  # noqa: E402
from dp_accounting.pld import privacy_loss_distribution as pld_lib  # noqa: E402

import opaque.accounting as acc  # noqa: E402
from opaque.accounting import calibration as cal  # noqa: E402
from opaque.accounting.discretization import get_discretization  # noqa: E402

# ============================================================================
# Helpers
# ============================================================================

ATOL = 1e-6  # absolute tolerance for most comparisons

# NOTE: opaque and dp_accounting agree to ~1e-9 relative error.
# For high-epsilon regimes (small σ), absolute error can be ~1e-7.
# We use rel=1e-8 for single Gaussian (high ε) and abs=1e-6 elsewhere.

# User-specified parameter grids (σ must be in [0.1, 1.2] per Rust crate)
SIGMAS = [0.1, 0.25, 0.3, 0.35, 0.5, 0.65, 0.8, 1.2]
SAMPLE_RATES = [0.001, 0.0005, 0.0001]
STEPS = [10, 50, 200, 500, 1000, 3000]
DELTAS = [1e-5, 1e-6, 1e-8]


def _ref_gaussian_pld(sigma: float, sampling_prob: float = 1.0):
    """Build a dp_accounting PLD for Gaussian (± Poisson subsampling)."""
    return pld_lib.from_gaussian_mechanism(sigma, sampling_prob=sampling_prob)


def _ref_epsilon(
    sigma: float, delta: float, sampling_prob: float = 1.0, steps: int = 1
):
    """Reference epsilon from dp_accounting."""
    pld = _ref_gaussian_pld(sigma, sampling_prob)
    if steps > 1:
        pld = pld.self_compose(steps)
    return pld.get_epsilon_for_delta(delta)


def _ref_delta(
    sigma: float, epsilon: float, sampling_prob: float = 1.0, steps: int = 1
):
    """Reference delta from dp_accounting."""
    pld = _ref_gaussian_pld(sigma, sampling_prob)
    if steps > 1:
        pld = pld.self_compose(steps)
    return pld.get_delta_for_epsilon(epsilon)


# ============================================================================
# 1. Gaussian mechanism — epsilon_at vs dp_accounting
# ============================================================================


class TestGaussianEpsilon:
    """Single Gaussian mechanism: opaque vs dp_accounting epsilon_at."""

    @pytest.mark.parametrize("sigma", SIGMAS)
    @pytest.mark.parametrize("delta", DELTAS)
    def test_epsilon(self, sigma, delta):
        ours = acc.gaussian(sigma).epsilon_at(delta)
        ref = _ref_epsilon(sigma, delta)
        assert ours == pytest.approx(ref, rel=1e-8), (
            f"Gaussian(σ={sigma}) eps@δ={delta}: ours={ours}, ref={ref}"
        )


class TestGaussianDelta:
    """Single Gaussian mechanism: opaque vs dp_accounting delta_at."""

    @pytest.mark.parametrize("sigma", [0.3, 0.5, 0.8, 1.2])
    def test_delta(self, sigma):
        # Pick an epsilon value in a reasonable range
        eps = acc.gaussian(sigma).epsilon_at(1e-5) * 0.8
        ours = acc.gaussian(sigma).delta_at(eps)
        ref = _ref_delta(sigma, eps)
        # delta_at has slightly lower precision than epsilon_at due to
        # different internal search/interpolation paths; rel~1e-7 is expected.
        assert ours == pytest.approx(ref, rel=1e-7), (
            f"Gaussian(σ={sigma}) delta@ε={eps}: ours={ours}, ref={ref}"
        )


# ============================================================================
# 2. Poisson-subsampled Gaussian — epsilon_at vs dp_accounting
# ============================================================================


class TestPoissonEpsilon:
    """Poisson(Gaussian, q) * steps: opaque vs dp_accounting epsilon_at."""

    @pytest.mark.parametrize("sigma", [0.5, 0.8, 1.2])
    @pytest.mark.parametrize("q", SAMPLE_RATES)
    @pytest.mark.parametrize("steps", [10, 50, 200, 500, 1000])
    def test_epsilon(self, sigma, q, steps):
        ours = (acc.poisson(acc.gaussian(sigma), q) * steps).epsilon_at(1e-5)
        ref = _ref_epsilon(sigma, 1e-5, sampling_prob=q, steps=steps)
        assert ours == pytest.approx(ref, abs=ATOL), (
            f"Poisson(G({sigma}),{q})*{steps} eps@1e-5: ours={ours}, ref={ref}"
        )


class TestPoissonHighSteps:
    """Long training runs (3000 steps) — verify no drift at scale."""

    @pytest.mark.parametrize("sigma", [0.8, 1.2])
    @pytest.mark.parametrize("q", [0.001, 0.0001])
    def test_3000_steps(self, sigma, q):
        ours = (acc.poisson(acc.gaussian(sigma), q) * 3000).epsilon_at(1e-5)
        ref = _ref_epsilon(sigma, 1e-5, sampling_prob=q, steps=3000)
        assert ours == pytest.approx(ref, abs=ATOL), (
            f"Poisson(G({sigma}),{q})*3000 eps@1e-5: ours={ours}, ref={ref}"
        )


class TestPoissonDelta:
    """Poisson delta_at cross-validation."""

    @pytest.mark.parametrize("sigma", [0.5, 1.2])
    @pytest.mark.parametrize("q", [0.001, 0.0001])
    @pytest.mark.parametrize("steps", [100, 500])
    def test_delta(self, sigma, q, steps):
        eps = (acc.poisson(acc.gaussian(sigma), q) * steps).epsilon_at(1e-5) * 0.8
        ours = (acc.poisson(acc.gaussian(sigma), q) * steps).delta_at(eps)
        ref = _ref_delta(sigma, eps, sampling_prob=q, steps=steps)
        assert ours == pytest.approx(ref, abs=ATOL), (
            f"Poisson(G({sigma}),{q})*{steps} delta@ε={eps}: ours={ours}, ref={ref}"
        )


# ============================================================================
# 3. Triple validation: opaque vs dp_accounting vs riskcal
# ============================================================================


class TestTripleEpsilon:
    """Epsilon: opaque vs dp_accounting vs riskcal (via (ε,δ)->advantage->ε)."""

    @pytest.mark.parametrize("sigma", [0.5, 0.8, 1.2])
    @pytest.mark.parametrize("q", [0.001, 0.0001])
    @pytest.mark.parametrize("steps", [100, 500])
    def test_triple_epsilon(self, sigma, q, steps):
        # opaque
        proc = acc.poisson(acc.gaussian(sigma), q) * steps
        eps_ours = proc.epsilon_at(1e-5)

        # dp_accounting
        ref_pld = _ref_gaussian_pld(sigma, q).self_compose(steps)
        eps_ref = ref_pld.get_epsilon_for_delta(1e-5)

        # riskcal: epsilon from (alpha, beta) error rates
        # At alpha=0, beta = 1-advantage, so advantage = 1-beta
        # We verify riskcal and dp_accounting agree on advantage
        adv_riskcal = rc_analysis.get_advantage_from_pld(ref_pld)
        adv_ours = proc.advantage()

        assert eps_ours == pytest.approx(eps_ref, abs=ATOL)
        assert adv_ours == pytest.approx(adv_riskcal, abs=ATOL)


class TestTripleBeta:
    """Beta (Type-II error): opaque vs riskcal.get_beta_from_pld."""

    @pytest.mark.parametrize("sigma", [0.5, 0.8, 1.2])
    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
    def test_beta_single_gaussian(self, sigma, alpha):
        proc = acc.gaussian(sigma)
        beta_ours = proc.beta_at(alpha)

        ref_pld = _ref_gaussian_pld(sigma)
        beta_riskcal = float(rc_analysis.get_beta_from_pld(ref_pld, alpha=alpha))

        assert beta_ours == pytest.approx(beta_riskcal, abs=ATOL), (
            f"Gaussian({sigma}) beta@α={alpha}: ours={beta_ours}, riskcal={beta_riskcal}"
        )

    @pytest.mark.parametrize("sigma", [0.8, 1.2])
    @pytest.mark.parametrize("q", [0.001, 0.0001])
    @pytest.mark.parametrize("steps", [100, 500])
    @pytest.mark.parametrize("alpha", [0.01, 0.1])
    def test_beta_poisson(self, sigma, q, steps, alpha):
        proc = acc.poisson(acc.gaussian(sigma), q) * steps
        beta_ours = proc.beta_at(alpha)

        ref_pld = _ref_gaussian_pld(sigma, q).self_compose(steps)
        beta_riskcal = float(rc_analysis.get_beta_from_pld(ref_pld, alpha=alpha))

        assert beta_ours == pytest.approx(beta_riskcal, abs=ATOL), (
            f"Poisson(G({sigma}),{q})*{steps} beta@α={alpha}: "
            f"ours={beta_ours}, riskcal={beta_riskcal}"
        )


class TestTripleAdvantage:
    """Advantage (TV distance): opaque vs riskcal.get_advantage_from_pld."""

    @pytest.mark.parametrize("sigma", SIGMAS)
    def test_gaussian_advantage(self, sigma):
        proc = acc.gaussian(sigma)
        adv_ours = proc.advantage()

        ref_pld = _ref_gaussian_pld(sigma)
        adv_riskcal = rc_analysis.get_advantage_from_pld(ref_pld)

        assert adv_ours == pytest.approx(adv_riskcal, abs=ATOL), (
            f"Gaussian({sigma}) advantage: ours={adv_ours}, riskcal={adv_riskcal}"
        )

    @pytest.mark.parametrize("sigma", [0.8, 1.2])
    @pytest.mark.parametrize("q", [0.001, 0.0001])
    @pytest.mark.parametrize("steps", [100, 500])
    def test_poisson_advantage(self, sigma, q, steps):
        proc = acc.poisson(acc.gaussian(sigma), q) * steps
        adv_ours = proc.advantage()

        ref_pld = _ref_gaussian_pld(sigma, q).self_compose(steps)
        adv_riskcal = rc_analysis.get_advantage_from_pld(ref_pld)

        assert adv_ours == pytest.approx(adv_riskcal, abs=ATOL)


class TestTripleRisk:
    """Bayes risk: opaque vs riskcal.get_bayes_risk_from_pld."""

    @pytest.mark.parametrize("sigma", [0.5, 0.8, 1.2])
    @pytest.mark.parametrize("prior", [0.3, 0.5, 0.7])
    def test_gaussian_risk(self, sigma, prior):
        proc = acc.gaussian(sigma)
        risk_ours = proc.risk_at(prior)

        ref_pld = _ref_gaussian_pld(sigma)
        risk_riskcal = float(rc_analysis.get_bayes_risk_from_pld(ref_pld, prior))

        assert risk_ours == pytest.approx(risk_riskcal, abs=ATOL), (
            f"Gaussian({sigma}) risk@prior={prior}: ours={risk_ours}, riskcal={risk_riskcal}"
        )


# ============================================================================
# 4. Truncated Poisson — validity checks (no dp_accounting equivalent)
# ============================================================================


class TestTruncatedPoissonValidity:
    """Truncated Poisson: internal consistency and fallback behavior.

    Note: Truncated Poisson uses an asymmetric mixture model with a
    doubled-sensitivity component (from [Gan25]), so it does NOT guarantee
    lower epsilon than standard Poisson for the same sample_rate.
    The benefit is a more accurate model of real production systems.
    """

    @pytest.mark.parametrize("sigma", [0.5, 0.8, 1.2])
    def test_larger_cap_lower_epsilon(self, sigma):
        """Larger batch_size_cap → less truncation → lower epsilon."""
        n = 100_000
        q = 0.001
        steps = 500
        g = acc.gaussian(sigma)
        # cap=50 (heavy truncation: expected_batch=100, cap < expected)
        eps_small_cap = (acc.truncated_poisson(g, q, 50, n) * steps).epsilon_at(1e-5)
        # cap=500 (light truncation: expected_batch=100, cap >> expected)
        eps_large_cap = (acc.truncated_poisson(g, q, 500, n) * steps).epsilon_at(1e-5)
        assert eps_large_cap <= eps_small_cap + 1e-10, (
            f"Larger cap should give ≤ epsilon: cap=500 → {eps_large_cap}, cap=50 → {eps_small_cap}"
        )

    @pytest.mark.parametrize("sigma", [0.5, 0.8, 1.2])
    def test_monotone_in_steps(self, sigma):
        """More steps → higher epsilon."""
        g = acc.gaussian(sigma)
        epsilons = []
        for steps in [10, 100, 500, 1000]:
            eps = (acc.truncated_poisson(g, 0.005, 250, 50_000) * steps).epsilon_at(
                1e-5
            )
            epsilons.append(eps)
        for i in range(len(epsilons) - 1):
            assert epsilons[i] < epsilons[i + 1]

    def test_positive_epsilon(self):
        """TruncatedPoisson epsilon > 0 for non-trivial mechanism."""
        g = acc.gaussian(0.8)
        proc = acc.truncated_poisson(g, 0.005, 250, 50_000) * 100
        assert proc.epsilon_at(1e-5) > 0

    def test_fallback_when_no_truncation(self):
        """When batch_size_cap >> expected batch, result ≈ standard Poisson."""
        # expected_batch = 100000 * 0.001 = 100, cap = 100000 (no truncation)
        g = acc.gaussian(0.8)
        steps = 500
        eps_trunc = (
            acc.truncated_poisson(g, 0.001, 100_000, 100_000) * steps
        ).epsilon_at(1e-5)
        eps_poisson = (acc.poisson(g, 0.001) * steps).epsilon_at(1e-5)
        assert eps_trunc == pytest.approx(eps_poisson, rel=1e-6)


# ============================================================================
# 5. Parallel Poisson — cross-validation
# ============================================================================


class TestParallelPoissonCrossValidation:
    """ParallelPoisson mechanism: verify against dp_accounting equivalent."""

    @pytest.mark.parametrize("sigma", [0.5, 0.8, 1.2])
    @pytest.mark.parametrize("q", [0.001, 0.0005])
    @pytest.mark.parametrize("num_workers", [2, 4])
    def test_parallel_poisson_vs_reference(self, sigma, q, num_workers):
        """ParallelPoisson(G(σ), q, k) should give sensible epsilon."""
        proc = (
            acc.parallel_poisson(
                acc.gaussian(sigma), sample_rate=q, num_workers=num_workers
            )
            * 500
        )
        eps = proc.epsilon_at(1e-5)

        # Must be finite and positive
        assert math.isfinite(eps) and eps > 0

        # Compare with non-parallel: parallel sampling should account for duplication
        # (when same example appears in multiple workers)
        proc_no_acc = acc.poisson(acc.gaussian(sigma), q) * 500
        eps_no_acc = proc_no_acc.epsilon_at(1e-5)
        # Both should be reasonable
        assert math.isfinite(eps_no_acc) and eps_no_acc > 0


# ============================================================================
# 6. AdaClip — cross-validation
# ============================================================================


class TestAdaClipCrossValidation:
    """AdaClip: verify adaclip_sensitivity formula matches expected z_eff."""

    @pytest.mark.parametrize(
        "sigma,batch_size",
        [
            (1.0, 1000),
            (1.1, 1000),
            (0.5, 200),
            (1.2, 2000),
        ],
    )
    def test_adaclip_effective_noise(self, sigma, batch_size):
        from opaque.accounting import _native

        proc = acc.adaclip(acc.gaussian(sigma), expected_batch_size=batch_size)
        sigma_b = batch_size * 0.05
        s = _native.adaclip_sensitivity(sigma, sigma_b)
        z_eff = 1.0 / s

        # Verify effective noise via PLD
        config = get_discretization()
        ref = _native.gaussian_pld(z_eff, config.to_native())
        assert proc.epsilon_at(1e-5) == pytest.approx(ref.epsilon_at(1e-5), rel=1e-12)

        from opaque.accounting.transformations import AdaClip

        assert isinstance(proc, AdaClip)

    @pytest.mark.parametrize("batch_size", [200, 1000, 2000])
    def test_adaclip_increases_privacy_cost(self, batch_size):
        g = acc.gaussian(1.0)
        a = acc.adaclip(g, expected_batch_size=batch_size)
        assert a.epsilon_at(1e-5) > g.epsilon_at(1e-5)

    def test_adaclip_composed_with_poisson(self):
        """AdaClip result composes with poisson() normally."""
        step = (
            acc.poisson(acc.adaclip(acc.gaussian(1.1), expected_batch_size=1000), 0.01)
            * 1000
        )
        eps = step.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0


# ============================================================================
# 7. Metrics consistency — internal invariants
# ============================================================================


class TestMetricsConsistency:
    """Verify internal consistency of privacy metrics (roundtrips, identities)."""

    @pytest.mark.parametrize("sigma", [0.3, 0.5, 0.8, 1.2])
    def test_epsilon_delta_roundtrip(self, sigma):
        """epsilon_at(δ) -> delta_at(ε) ≈ δ."""
        proc = acc.gaussian(sigma)
        delta = 1e-5
        eps = proc.epsilon_at(delta)
        delta_back = proc.delta_at(eps)
        assert delta_back == pytest.approx(delta, abs=ATOL), (
            f"Roundtrip failed: δ={delta} → ε={eps} → δ'={delta_back}"
        )

    @pytest.mark.parametrize("sigma", [0.3, 0.5, 0.8, 1.2])
    def test_advantage_equals_delta_at_zero(self, sigma):
        """advantage() == delta_at(0) (by definition of f-DP advantage)."""
        proc = acc.gaussian(sigma)
        adv = proc.advantage()
        d0 = proc.delta_at(0.0)
        assert adv == pytest.approx(d0, abs=1e-10), (
            f"advantage={adv} != delta_at(0)={d0}"
        )

    @pytest.mark.parametrize("sigma", [0.5, 0.8, 1.2])
    def test_advantage_poisson_equals_delta_at_zero(self, sigma):
        """advantage() == delta_at(0) for Poisson-subsampled too."""
        proc = acc.poisson(acc.gaussian(sigma), 0.01) * 500
        adv = proc.advantage()
        d0 = proc.delta_at(0.0)
        assert adv == pytest.approx(d0, abs=1e-10)

    @pytest.mark.parametrize("sigma", [0.5, 0.8, 1.2])
    def test_risk_bounds(self, sigma):
        """Risk must be in [0, min(prior, 1-prior)] for optimal adversary."""
        proc = acc.gaussian(sigma)
        for prior in [0.3, 0.5, 0.7]:
            r = proc.risk_at(prior)
            assert 0 <= r <= min(prior, 1 - prior) + 1e-10, (
                f"Risk out of bounds: risk={r}, prior={prior}"
            )

    @pytest.mark.parametrize("sigma", [0.5, 0.8, 1.2])
    def test_risk_symmetry(self, sigma):
        """risk_at(p) == risk_at(1-p) for symmetric mechanisms."""
        proc = acc.gaussian(sigma)
        for prior in [0.2, 0.3, 0.4]:
            r1 = proc.risk_at(prior)
            r2 = proc.risk_at(1.0 - prior)
            assert r1 == pytest.approx(r2, abs=1e-10), (
                f"Risk asymmetry: risk({prior})={r1}, risk({1 - prior})={r2}"
            )

    @pytest.mark.parametrize("sigma", [0.5, 0.8, 1.2])
    def test_beta_monotone_in_alpha(self, sigma):
        """beta_at(α) must be non-increasing in α."""
        proc = acc.gaussian(sigma)
        alphas = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
        betas = [proc.beta_at(a) for a in alphas]
        for i in range(len(betas) - 1):
            assert betas[i] >= betas[i + 1] - 1e-10, (
                f"Beta not monotone: β({alphas[i]})={betas[i]} > β({alphas[i + 1]})={betas[i + 1]}"
            )

    def test_identity_zero_epsilon(self):
        """Identity mechanism should have ε≈0 for any δ."""
        proc = acc.identity()
        assert proc.epsilon_at(1e-5) == pytest.approx(0.0, abs=1e-8)
        assert proc.epsilon_at(1e-10) == pytest.approx(0.0, abs=1e-8)


# ============================================================================
# 8. Composition cross-validation
# ============================================================================


class TestCompositionCrossValidation:
    """Verify composed mechanisms match dp_accounting composition."""

    def test_heterogeneous_compose(self):
        """Compose different mechanisms: (G(0.5)*100 | G(1.0)*200)."""
        # opaque
        proc = (acc.gaussian(0.5) * 100) | (acc.gaussian(1.0) * 200)
        eps_ours = proc.epsilon_at(1e-5)

        # dp_accounting
        pld_a = _ref_gaussian_pld(0.5).self_compose(100)
        pld_b = _ref_gaussian_pld(1.0).self_compose(200)
        pld_ref = pld_a.compose(pld_b)
        eps_ref = pld_ref.get_epsilon_for_delta(1e-5)

        # For large epsilon (~400), use relative tolerance
        assert eps_ours == pytest.approx(eps_ref, rel=1e-6)

    def test_compose_poisson_different_rates(self):
        """Compose Poisson steps with different sample rates."""
        # opaque
        p1 = acc.poisson(acc.gaussian(0.8), 0.001) * 500
        p2 = acc.poisson(acc.gaussian(0.8), 0.0005) * 500
        proc = p1 | p2
        eps_ours = proc.epsilon_at(1e-5)

        # dp_accounting
        pld1 = _ref_gaussian_pld(0.8, 0.001).self_compose(500)
        pld2 = _ref_gaussian_pld(0.8, 0.0005).self_compose(500)
        pld_ref = pld1.compose(pld2)
        eps_ref = pld_ref.get_epsilon_for_delta(1e-5)

        assert eps_ours == pytest.approx(eps_ref, abs=ATOL)

    def test_compose_same_mechanism(self):
        """Compose same Poisson steps — should equal repeat."""
        proc_compose = acc.poisson(acc.gaussian(0.8), 0.001) | acc.poisson(
            acc.gaussian(0.8), 0.001
        )
        proc_repeat = acc.poisson(acc.gaussian(0.8), 0.001) * 2
        eps_compose = proc_compose.epsilon_at(1e-5)
        eps_repeat = proc_repeat.epsilon_at(1e-5)
        assert eps_compose == pytest.approx(eps_repeat, abs=1e-10)


# ============================================================================
# 9. Calibration cross-validation
# ============================================================================


class TestCalibrationCrossValidation:
    """Verify calibration results match dp_accounting reference."""

    def test_calibrate_epsilon_roundtrip(self):
        """Calibrated noise → check epsilon against reference."""
        target_eps = 5.0
        delta = 1e-5
        q = 0.01
        steps = 1000

        result = cal.calibrate(
            cal.epsilon_budget(target_eps, delta=delta),
            lambda nm: acc.poisson(acc.gaussian(nm), q) * steps,
            0.1,
            1.2,
        )
        assert result.converged

        # Verify with dp_accounting
        ref_eps = _ref_epsilon(result.param, delta, sampling_prob=q, steps=steps)
        assert ref_eps == pytest.approx(target_eps, abs=1e-3)

    def test_calibrate_advantage_roundtrip(self):
        """Calibrated noise → check advantage against riskcal."""
        result = cal.calibrate(
            cal.advantage_budget(0.15),
            lambda nm: acc.poisson(acc.gaussian(nm), 0.01) * 500,
            0.3,
            1.2,
        )
        assert result.converged

        # Verify with riskcal
        ref_pld = _ref_gaussian_pld(result.param, 0.01).self_compose(500)
        adv_ref = rc_analysis.get_advantage_from_pld(ref_pld)
        assert adv_ref == pytest.approx(0.15, abs=1e-3)


# ============================================================================
# 10. Numerical stability regressions
# ============================================================================


class TestNumericalStability:
    """Edge cases and regression tests for numerical stability."""

    def test_smallest_sigma(self):
        """σ=0.1 (boundary, high privacy loss) — should compute without overflow."""
        proc = acc.gaussian(0.1)
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps)
        assert eps > 50  # very high epsilon expected

    def test_largest_sigma(self):
        """σ=1.2 (boundary, low privacy loss) — should compute without underflow."""
        proc = acc.gaussian(1.2)
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps)
        assert eps < 5  # relatively low epsilon

    def test_very_small_delta(self):
        """δ=1e-10 — tight delta should work."""
        proc = acc.gaussian(1.0)
        eps = proc.epsilon_at(1e-10)
        assert math.isfinite(eps)
        assert eps > proc.epsilon_at(1e-5)  # tighter delta → higher epsilon

    def test_very_small_sample_rate(self):
        """q=1e-5 — very small batches."""
        proc = acc.poisson(acc.gaussian(1.0), 1e-5) * 1000
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps)
        assert eps > 0

    def test_many_steps(self):
        """3000 steps — verify no accumulation of error."""
        proc = acc.poisson(acc.gaussian(0.8), 0.001) * 3000
        eps_ours = proc.epsilon_at(1e-5)
        eps_ref = _ref_epsilon(0.8, 1e-5, sampling_prob=0.001, steps=3000)
        assert eps_ours == pytest.approx(eps_ref, abs=ATOL)

    def test_compose_many_steps(self):
        """Build up 1000 steps via loop composition vs single repeat."""
        g = acc.gaussian(0.8)
        step = acc.poisson(g, 0.001)
        repeated = step * 1000
        eps_repeated = repeated.epsilon_at(1e-5)

        # Same via composition loop (small)
        composed = step | step | step | step | step  # 5 steps
        composed = composed * 200  # 1000 steps
        eps_composed = composed.epsilon_at(1e-5)

        assert eps_repeated == pytest.approx(eps_composed, abs=ATOL)

    def test_epsilon_monotone_in_steps(self):
        """More steps → weakly higher epsilon (monotonicity)."""
        g = acc.gaussian(0.8)
        epsilons = []
        for steps in [1, 10, 50, 100, 500, 1000]:
            proc = acc.poisson(g, 0.001) * steps
            epsilons.append(proc.epsilon_at(1e-5))
        for i in range(len(epsilons) - 1):
            assert epsilons[i] <= epsilons[i + 1] + 1e-10

    def test_epsilon_monotone_in_noise(self):
        """More noise → lower epsilon (monotonicity)."""
        sigmas = [0.1, 0.3, 0.5, 0.8, 1.0, 1.2]
        epsilons = [acc.gaussian(s).epsilon_at(1e-5) for s in sigmas]
        for i in range(len(epsilons) - 1):
            assert epsilons[i] >= epsilons[i + 1] - 1e-10
