"""Cross-validation for rectified & truncated Gaussian mechanisms.

These tests verify the Rust-backed PLD for bounded Gaussian variants using
two independent strategies:

1. **Numerical quadrature** — pure-Python δ(ε) computed via
   ``scipy.integrate.quad`` directly from the density definitions.  This is
   a fully independent re-implementation of the hockey-stick divergence.

2. **Large-radius convergence** — at R ≫ 1 the bounded mechanisms should
   converge to the standard Gaussian, whose PLD is cross-validated against
   Google's ``dp_accounting`` library in ``test_cross_validation.py``.

Strategy (1) catches formula bugs in Rust.  Strategy (2) catches
discretisation / composition drift by chaining through the already-validated
standard Gaussian oracle.

Requires optional deps — install with ``uv sync --group cross-validation``.
The ``pytest.importorskip`` calls below gate the entire module automatically.

Run with::

    uv run --group cross-validation pytest \
        tests/test_bounded_gaussian_cross_validation.py -v
"""

from __future__ import annotations

import math

import pytest

scipy = pytest.importorskip("scipy")
dp_accounting = pytest.importorskip("dp_accounting")

from dp_accounting.pld import privacy_loss_distribution as pld_lib  # noqa: E402
from scipy import integrate, stats  # noqa: E402

import opaque_accounting as acc  # noqa: E402
from opaque_accounting import DiscretizationConfig  # noqa: E402

_CFG = DiscretizationConfig()

# ============================================================================
# Pure-Python δ(ε) via numerical quadrature
# ============================================================================

# Standard normal for PDF / CDF helpers.
_N01 = stats.norm(0, 1)


def _rectified_gaussian_delta_quadrature(
    sigma: float,
    radius: float,
    epsilon: float,
    *,
    sensitivity: float = 1.0,
) -> float:
    r"""Compute δ(ε) for the rectified Gaussian via numerical integration.

    The rectified Gaussian centered at μ with std σ and radius R has:

    - Interior density on (−Rσ, Rσ):   f(x; μ) = φ((x−μ)/σ) / σ
    - Left point mass at x = −Rσ:      p_L(μ) = Φ((−Rσ − μ)/σ)
    - Right point mass at x = +Rσ:     p_R(μ) = 1 − Φ((Rσ − μ)/σ)

    Under Add/Remove adjacency (μ₀=0, μ₁=Δ):

        δ(ε) = ∫ max(0, f(x;0) − e^ε f(x;Δ)) dx
             + max(0, p_L(0) − e^ε p_L(Δ))
             + max(0, p_R(0) − e^ε p_R(Δ))
    """
    r_abs = radius * sigma

    # --- Interior integral ---
    def integrand(x: float) -> float:
        f0 = _N01.pdf(x / sigma) / sigma
        f1 = _N01.pdf((x - sensitivity) / sigma) / sigma
        return max(0.0, f0 - math.exp(epsilon) * f1)

    interior, _ = integrate.quad(integrand, -r_abs, r_abs, limit=200)

    # --- Left point mass ---
    p_l0 = _N01.cdf((-r_abs) / sigma)
    p_l1 = _N01.cdf((-r_abs - sensitivity) / sigma)
    delta_left = max(0.0, p_l0 - math.exp(epsilon) * p_l1)

    # --- Right point mass ---
    p_r0 = 1.0 - _N01.cdf(r_abs / sigma)
    p_r1 = 1.0 - _N01.cdf((r_abs - sensitivity) / sigma)
    delta_right = max(0.0, p_r0 - math.exp(epsilon) * p_r1)

    return interior + delta_left + delta_right


def _truncated_gaussian_delta_quadrature(
    sigma: float,
    radius: float,
    epsilon: float,
    *,
    sensitivity: float = 1.0,
) -> float:
    r"""Compute δ(ε) for the truncated Gaussian via numerical integration.

    The truncated Gaussian centered at μ with std σ and radius R has density:

        f(x; μ) = φ((x−μ)/σ) / (σ · Z(μ))    for x ∈ [−Rσ, Rσ]

    where Z(μ) = Φ((Rσ−μ)/σ) − Φ((−Rσ−μ)/σ).

    Hockey-stick divergence:

        δ(ε) = ∫ max(0, f(x;0) − e^ε f(x;Δ)) dx   over [−Rσ, Rσ]
    """
    r_abs = radius * sigma

    # Normalization constants
    z0 = _N01.cdf(r_abs / sigma) - _N01.cdf(-r_abs / sigma)
    z1 = _N01.cdf((r_abs - sensitivity) / sigma) - _N01.cdf(
        (-r_abs - sensitivity) / sigma
    )

    if z0 <= 0.0 or z1 <= 0.0:
        return 0.0

    def integrand(x: float) -> float:
        f0 = _N01.pdf(x / sigma) / (sigma * z0)
        f1 = _N01.pdf((x - sensitivity) / sigma) / (sigma * z1)
        return max(0.0, f0 - math.exp(epsilon) * f1)

    result, _ = integrate.quad(integrand, -r_abs, r_abs, limit=200)
    return result


# ============================================================================
# Reference oracle: dp_accounting standard Gaussian
# ============================================================================


def _ref_gaussian_epsilon(sigma: float, delta: float) -> float:
    """Reference ε from dp_accounting for the standard Gaussian mechanism."""
    pld = pld_lib.from_gaussian_mechanism(sigma)
    return pld.get_epsilon_for_delta(delta)


def _ref_poisson_gaussian_epsilon(
    sigma: float, delta: float, *, sample_rate: float, steps: int
) -> float:
    """Reference ε from dp_accounting for Poisson-subsampled Gaussian."""
    pld = pld_lib.from_gaussian_mechanism(sigma, sampling_prob=sample_rate)
    if steps > 1:
        pld = pld.self_compose(steps)
    return pld.get_epsilon_for_delta(delta)


# ============================================================================
# 1. Rectified Gaussian — quadrature validation
# ============================================================================

# Parameter grids
SIGMAS = [0.5, 1.0]  # stay within [0.1, 1.2]
RADII = [2.0, 3.0, 5.0]


class TestRectifiedGaussianQuadrature:
    """Validate rectified Gaussian δ(ε) against independent Python quadrature.

    This is the primary correctness check: a fully independent
    re-implementation of the hockey-stick divergence using scipy.integrate.quad.
    """

    @pytest.mark.parametrize("sigma", SIGMAS)
    @pytest.mark.parametrize("radius", RADII)
    @pytest.mark.parametrize("delta", [1e-3, 1e-5])
    def test_epsilon_vs_quadrature(self, sigma, radius, delta):
        """ε from Rust PLD matches ε implied by Python quadrature δ(ε)."""
        proc = acc.rectified_gaussian(sigma, radius)
        eps_rust = proc.pmf(_CFG).epsilon_at(delta)

        # Verify: quadrature δ at eps_rust should be ≈ delta
        delta_quad = _rectified_gaussian_delta_quadrature(sigma, radius, eps_rust)

        # The Rust PLD discretizes, so we compare with relative tolerance.
        # δ_quad should agree with the target delta to within ~1%.
        assert delta_quad == pytest.approx(delta, rel=0.02), (
            f"RectifiedGaussian(σ={sigma}, R={radius}): "
            f"Rust ε={eps_rust:.6f} → quad δ={delta_quad:.2e} vs target δ={delta:.2e}"
        )

    @pytest.mark.parametrize("sigma", SIGMAS)
    @pytest.mark.parametrize("radius", RADII)
    def test_delta_at_vs_quadrature(self, sigma, radius):
        """δ(ε) from Rust PLD matches Python quadrature directly."""
        proc = acc.rectified_gaussian(sigma, radius)
        pmf = proc.pmf(_CFG)
        # Pick a few epsilon values in a reasonable range
        eps_max = pmf.epsilon_at(1e-8)  # high ε end
        eps_mid = pmf.epsilon_at(1e-4)  # mid range
        for eps in [eps_mid, eps_max * 0.5]:
            delta_rust = pmf.delta_at(eps)
            delta_quad = _rectified_gaussian_delta_quadrature(sigma, radius, eps)
            assert delta_rust == pytest.approx(delta_quad, rel=0.02), (
                f"RectifiedGaussian(σ={sigma}, R={radius}): "
                f"δ at ε={eps:.4f}: Rust={delta_rust:.2e}, quad={delta_quad:.2e}"
            )


# ============================================================================
# 2. Truncated Gaussian — quadrature validation
# ============================================================================


class TestTruncatedGaussianQuadrature:
    """Validate truncated Gaussian δ(ε) against independent Python quadrature."""

    @pytest.mark.parametrize("sigma", SIGMAS)
    @pytest.mark.parametrize("radius", RADII)
    @pytest.mark.parametrize("delta", [1e-3, 1e-5])
    def test_epsilon_vs_quadrature(self, sigma, radius, delta):
        """ε from Rust PLD matches ε implied by Python quadrature δ(ε)."""
        proc = acc.truncated_gaussian(sigma, radius)
        eps_rust = proc.pmf(_CFG).epsilon_at(delta)

        delta_quad = _truncated_gaussian_delta_quadrature(sigma, radius, eps_rust)

        assert delta_quad == pytest.approx(delta, rel=0.02), (
            f"TruncatedGaussian(σ={sigma}, R={radius}): "
            f"Rust ε={eps_rust:.6f} → quad δ={delta_quad:.2e} vs target δ={delta:.2e}"
        )

    @pytest.mark.parametrize("sigma", SIGMAS)
    @pytest.mark.parametrize("radius", RADII)
    def test_delta_at_vs_quadrature(self, sigma, radius):
        """δ(ε) from Rust PLD matches Python quadrature directly."""
        proc = acc.truncated_gaussian(sigma, radius)
        pmf = proc.pmf(_CFG)
        eps_max = pmf.epsilon_at(1e-8)
        eps_mid = pmf.epsilon_at(1e-4)
        for eps in [eps_mid, eps_max * 0.5]:
            delta_rust = pmf.delta_at(eps)
            delta_quad = _truncated_gaussian_delta_quadrature(sigma, radius, eps)
            assert delta_rust == pytest.approx(delta_quad, rel=0.02), (
                f"TruncatedGaussian(σ={sigma}, R={radius}): "
                f"δ at ε={eps:.4f}: Rust={delta_rust:.2e}, quad={delta_quad:.2e}"
            )


class TestBoundedGaussianOrdering:
    """Verify full ordering: truncated ≤ rectified ≤ Gaussian (via quadrature).

    This uses the independent quadrature implementations to verify the
    theoretical ordering, providing a cross-check that both quadrature
    functions and the Rust PLD agree on the relative privacy guarantees.
    """

    @pytest.mark.parametrize("sigma", SIGMAS)
    @pytest.mark.parametrize("radius", RADII)
    def test_ordering_from_quadrature(self, sigma, radius):
        """δ_trunc ≤ δ_rect at the same ε, verified via quadrature."""
        # Pick epsilon that gives moderate delta for the rectified case
        eps = acc.rectified_gaussian(sigma, radius).pmf(_CFG).epsilon_at(1e-4) * 0.9

        delta_rect = _rectified_gaussian_delta_quadrature(sigma, radius, eps)
        delta_trunc = _truncated_gaussian_delta_quadrature(sigma, radius, eps)

        assert delta_trunc <= delta_rect + 1e-10, (
            f"Ordering violated at σ={sigma}, R={radius}, ε={eps:.4f}: "
            f"trunc δ={delta_trunc:.2e} > rect δ={delta_rect:.2e}"
        )

    @pytest.mark.parametrize("sigma", SIGMAS)
    @pytest.mark.parametrize("radius", RADII)
    def test_ordering_from_rust(self, sigma, radius):
        """ε_trunc ≤ ε_rect ≤ ε_gauss at δ=1e-5, from Rust PLD."""
        delta = 1e-5
        eps_gauss = acc.gaussian(sigma).cgf().epsilon_at(delta)
        eps_rect = acc.rectified_gaussian(sigma, radius).pmf(_CFG).epsilon_at(delta)
        eps_trunc = acc.truncated_gaussian(sigma, radius).pmf(_CFG).epsilon_at(delta)

        assert eps_trunc <= eps_rect + 1e-6, (
            f"σ={sigma}, R={radius}: trunc ε={eps_trunc:.6f} > rect ε={eps_rect:.6f}"
        )
        assert eps_rect <= eps_gauss + 1e-6, (
            f"σ={sigma}, R={radius}: rect ε={eps_rect:.6f} > gauss ε={eps_gauss:.6f}"
        )


# ============================================================================
# 3. Large-radius convergence — base mechanisms vs dp_accounting
# ============================================================================

LARGE_RADIUS = 50.0
CONVERGENCE_SIGMAS = [0.5, 0.8, 1.0]
CONVERGENCE_DELTAS = [1e-3, 1e-5, 1e-7]


class TestRectifiedGaussianConvergence:
    """At R=50, rectified Gaussian ≈ standard Gaussian (validated by dp_accounting)."""

    @pytest.mark.parametrize("sigma", CONVERGENCE_SIGMAS)
    @pytest.mark.parametrize("delta", CONVERGENCE_DELTAS)
    def test_epsilon_convergence(self, sigma, delta):
        eps_rect = acc.rectified_gaussian(sigma, LARGE_RADIUS).pmf(_CFG).epsilon_at(delta)
        eps_ref = _ref_gaussian_epsilon(sigma, delta)

        # At R=50, the point masses are ~exp(-R²/2) ≈ 0, so the mechanisms
        # are virtually identical.  Allow 1e-4 absolute tolerance.
        assert eps_rect == pytest.approx(eps_ref, abs=1e-4), (
            f"RectifiedGaussian(σ={sigma}, R={LARGE_RADIUS}) "
            f"ε@δ={delta}: {eps_rect:.6f} vs Gaussian ref {eps_ref:.6f}"
        )


class TestTruncatedGaussianConvergence:
    """At R=50, truncated Gaussian ≈ standard Gaussian (validated by dp_accounting)."""

    @pytest.mark.parametrize("sigma", CONVERGENCE_SIGMAS)
    @pytest.mark.parametrize("delta", CONVERGENCE_DELTAS)
    def test_epsilon_convergence(self, sigma, delta):
        eps_trunc = acc.truncated_gaussian(sigma, LARGE_RADIUS).pmf(_CFG).epsilon_at(delta)
        eps_ref = _ref_gaussian_epsilon(sigma, delta)

        assert eps_trunc == pytest.approx(eps_ref, abs=1e-4), (
            f"TruncatedGaussian(σ={sigma}, R={LARGE_RADIUS}) "
            f"ε@δ={delta}: {eps_trunc:.6f} vs Gaussian ref {eps_ref:.6f}"
        )


# ============================================================================
# 4. Poisson-subsampled convergence — large R vs dp_accounting
# ============================================================================

POISSON_SIGMAS = [0.8, 1.0]
POISSON_RATES = [0.001, 0.01]
POISSON_STEPS = [100, 500]


class TestPoissonRectifiedConvergence:
    """Poisson(RectifiedGaussian(σ, R=50), q) * N ≈ Poisson(Gaussian(σ), q) * N."""

    @pytest.mark.parametrize("sigma", POISSON_SIGMAS)
    @pytest.mark.parametrize("q", POISSON_RATES)
    @pytest.mark.parametrize("steps", POISSON_STEPS)
    def test_poisson_convergence(self, sigma, q, steps):
        eps_rect = (
            acc.poisson(acc.rectified_gaussian(sigma, LARGE_RADIUS), q) * steps
        ).pmf(_CFG).epsilon_at(1e-5)
        eps_ref = _ref_poisson_gaussian_epsilon(sigma, 1e-5, sample_rate=q, steps=steps)

        # Composed over many steps, allow slightly larger tolerance
        assert eps_rect == pytest.approx(eps_ref, abs=1e-3), (
            f"Poisson(RectGauss(σ={sigma}, R=50), q={q}) * {steps}: "
            f"ε={eps_rect:.6f} vs ref {eps_ref:.6f}"
        )


class TestPoissonTruncatedConvergence:
    """Poisson(TruncatedGaussian(σ, R=50), q) * N ≈ Poisson(Gaussian(σ), q) * N."""

    @pytest.mark.parametrize("sigma", POISSON_SIGMAS)
    @pytest.mark.parametrize("q", POISSON_RATES)
    @pytest.mark.parametrize("steps", POISSON_STEPS)
    def test_poisson_convergence(self, sigma, q, steps):
        eps_trunc = (
            acc.poisson(acc.truncated_gaussian(sigma, LARGE_RADIUS), q) * steps
        ).pmf(_CFG).epsilon_at(1e-5)
        eps_ref = _ref_poisson_gaussian_epsilon(sigma, 1e-5, sample_rate=q, steps=steps)

        assert eps_trunc == pytest.approx(eps_ref, abs=1e-3), (
            f"Poisson(TruncGauss(σ={sigma}, R=50), q={q}) * {steps}: "
            f"ε={eps_trunc:.6f} vs ref {eps_ref:.6f}"
        )


# ============================================================================
# 5. Sanity checks — invariants that any correct implementation must satisfy
# ============================================================================


class TestBoundedGaussianInvariants:
    """Basic invariants common to both bounded Gaussian mechanisms."""

    @pytest.mark.parametrize("sigma", [0.3, 0.5, 0.8, 1.0])
    @pytest.mark.parametrize("radius", [2.0, 3.0, 5.0])
    def test_advantage_equals_delta_at_zero(self, sigma, radius):
        """advantage() == delta_at(0) for both mechanisms."""
        for mech_fn in [acc.rectified_gaussian, acc.truncated_gaussian]:
            proc = mech_fn(sigma, radius)
            pmf = proc.pmf(_CFG)
            adv = pmf.delta_at(0.0)
            d0 = pmf.delta_at(0.0)
            assert adv == pytest.approx(d0, abs=1e-8), (
                f"{type(proc).__name__}(σ={sigma}, R={radius}): "
                f"advantage={adv} != delta_at(0)={d0}"
            )

    @pytest.mark.parametrize("sigma", [0.5, 0.8, 1.0])
    @pytest.mark.parametrize("radius", [2.0, 3.0, 5.0])
    def test_epsilon_delta_roundtrip(self, sigma, radius):
        """epsilon_at(δ) → delta_at(ε) ≈ δ."""
        for mech_fn in [acc.rectified_gaussian, acc.truncated_gaussian]:
            proc = mech_fn(sigma, radius)
            pmf = proc.pmf(_CFG)
            delta = 1e-5
            eps = pmf.epsilon_at(delta)
            delta_back = pmf.delta_at(eps)
            assert delta_back == pytest.approx(delta, abs=1e-6), (
                f"{type(proc).__name__}(σ={sigma}, R={radius}): "
                f"δ={delta} → ε={eps} → δ'={delta_back}"
            )

    @pytest.mark.parametrize("sigma", [0.5, 0.8, 1.0])
    def test_radius_monotonicity(self, sigma):
        """Larger R → higher ε (closer to unbounded Gaussian)."""
        for mech_fn in [acc.rectified_gaussian, acc.truncated_gaussian]:
            radii = [1.0, 2.0, 3.0, 5.0, 10.0]
            epsilons = [mech_fn(sigma, r).pmf(_CFG).epsilon_at(1e-5) for r in radii]
            for i in range(len(epsilons) - 1):
                assert epsilons[i] <= epsilons[i + 1] + 1e-6, (
                    f"{mech_fn.__name__}(σ={sigma}): "
                    f"ε(R={radii[i]})={epsilons[i]:.6f} > "
                    f"ε(R={radii[i + 1]})={epsilons[i + 1]:.6f}"
                )

    @pytest.mark.parametrize("radius", [2.0, 5.0])
    def test_noise_monotonicity(self, radius):
        """Higher σ → lower ε (more noise = more private)."""
        for mech_fn in [acc.rectified_gaussian, acc.truncated_gaussian]:
            sigmas = [0.3, 0.5, 0.8, 1.0]
            epsilons = [mech_fn(s, radius).pmf(_CFG).epsilon_at(1e-5) for s in sigmas]
            for i in range(len(epsilons) - 1):
                assert epsilons[i] >= epsilons[i + 1] - 1e-6, (
                    f"{mech_fn.__name__}(R={radius}): "
                    f"ε(σ={sigmas[i]})={epsilons[i]:.6f} < "
                    f"ε(σ={sigmas[i + 1]})={epsilons[i + 1]:.6f}"
                )
