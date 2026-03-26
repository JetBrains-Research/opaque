"""Cross-validation for truncated Gaussian mechanism.

These tests verify the Rust-backed PLD for the truncated Gaussian using
two independent strategies:

1. **Numerical quadrature** — pure-Python δ(ε) computed via
   ``scipy.integrate.quad`` directly from the density definitions.  This is
   a fully independent re-implementation of the hockey-stick divergence,
   including worst-case center optimization.

2. **Large-radius convergence** — at R ≫ 1 the truncated mechanism should
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

# ============================================================================
# Pure-Python δ(ε) via numerical quadrature
# ============================================================================

# Standard normal for PDF / CDF helpers.
_N01 = stats.norm(0, 1)


def _truncated_gaussian_delta_quadrature(
    sigma: float,
    radius: float,
    epsilon: float,
    *,
    sensitivity: float = 1.0,
    mu0: float = 0.0,
) -> float:
    r"""Compute δ(ε) for the truncated Gaussian via numerical integration.

    The truncated Gaussian centered at μ with std σ and radius R has density:

        f(x; μ) = φ((x−μ)/σ) / (σ · Z(μ))    for x ∈ [−Rσ, Rσ]

    where Z(μ) = Φ((Rσ−μ)/σ) − Φ((−Rσ−μ)/σ).

    Hockey-stick divergence:

        δ(ε) = ∫ max(0, f(x;μ₀) − e^ε f(x;μ₁)) dx   over [−Rσ, Rσ]
    """
    r_abs = radius * sigma
    mu1 = mu0 + sensitivity

    # Normalization constants
    z0 = _N01.cdf((r_abs - mu0) / sigma) - _N01.cdf((-r_abs - mu0) / sigma)
    z1 = _N01.cdf((r_abs - mu1) / sigma) - _N01.cdf((-r_abs - mu1) / sigma)

    if z0 <= 0.0 or z1 <= 0.0:
        return 0.0

    def integrand(x: float) -> float:
        f0 = _N01.pdf((x - mu0) / sigma) / (sigma * z0)
        f1 = _N01.pdf((x - mu1) / sigma) / (sigma * z1)
        return max(0.0, f0 - math.exp(epsilon) * f1)

    result, _ = integrate.quad(integrand, -r_abs, r_abs, limit=200)
    return result


def _truncated_gaussian_delta_worst_case(
    sigma: float,
    radius: float,
    epsilon: float,
    *,
    sensitivity: float = 1.0,
    n_centers: int = 200,
) -> float:
    """Worst-case δ(ε) over clipped input centers (matches Rust grid search)."""
    # After L2 clipping to norm Δ, per-coordinate inputs are in [-Δ, Δ].
    lo = -sensitivity
    hi = sensitivity
    worst = 0.0
    for i in range(n_centers + 1):
        mu0 = lo + (hi - lo) * i / n_centers
        d = _truncated_gaussian_delta_quadrature(
            sigma, radius, epsilon, sensitivity=sensitivity, mu0=mu0,
        )
        worst = max(worst, d)
    return worst


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
# 1. Truncated Gaussian — quadrature validation
# ============================================================================

# Parameter grids
SIGMAS = [0.5, 1.0]  # stay within [0.1, 1.2]
RADII = [2.0, 3.0, 5.0]


class TestTruncatedGaussianQuadrature:
    """Validate truncated Gaussian δ(ε) against independent Python quadrature."""

    @pytest.mark.parametrize("sigma", SIGMAS)
    @pytest.mark.parametrize("radius", RADII)
    @pytest.mark.parametrize("delta", [1e-3, 1e-5])
    def test_epsilon_vs_quadrature(self, sigma, radius, delta):
        """ε from Rust PLD matches ε implied by Python quadrature δ(ε)."""
        proc = acc.truncated_gaussian(sigma, radius)
        eps_rust = proc.epsilon_at(delta)

        delta_quad = _truncated_gaussian_delta_worst_case(sigma, radius, eps_rust)

        assert delta_quad == pytest.approx(delta, rel=0.02), (
            f"TruncatedGaussian(σ={sigma}, R={radius}): "
            f"Rust ε={eps_rust:.6f} → quad δ={delta_quad:.2e} vs target δ={delta:.2e}"
        )

    @pytest.mark.parametrize("sigma", SIGMAS)
    @pytest.mark.parametrize("radius", RADII)
    def test_delta_at_vs_quadrature(self, sigma, radius):
        """δ(ε) from Rust PLD matches Python quadrature directly."""
        proc = acc.truncated_gaussian(sigma, radius)
        eps_max = proc.epsilon_at(1e-8)
        eps_mid = proc.epsilon_at(1e-4)
        for eps in [eps_mid, eps_max * 0.5]:
            delta_rust = proc.delta_at(eps)
            delta_quad = _truncated_gaussian_delta_worst_case(sigma, radius, eps)
            assert delta_rust == pytest.approx(delta_quad, rel=0.02), (
                f"TruncatedGaussian(σ={sigma}, R={radius}): "
                f"δ at ε={eps:.4f}: Rust={delta_rust:.2e}, quad={delta_quad:.2e}"
            )


# ============================================================================
# 2. Large-radius convergence — base mechanisms vs dp_accounting
# ============================================================================

LARGE_RADIUS = 50.0
CONVERGENCE_SIGMAS = [0.5, 0.8, 1.0]
CONVERGENCE_DELTAS = [1e-3, 1e-5, 1e-7]


class TestTruncatedGaussianConvergence:
    """At R=50, truncated Gaussian ≈ standard Gaussian (validated by dp_accounting).

    With centers restricted to [-Δ, Δ] (clipped input domain), the
    normalization constants are ≈1 at R=50, so the gap is negligible.
    """

    @pytest.mark.parametrize("sigma", CONVERGENCE_SIGMAS)
    @pytest.mark.parametrize("delta", CONVERGENCE_DELTAS)
    def test_epsilon_convergence(self, sigma, delta):
        eps_trunc = acc.truncated_gaussian(sigma, LARGE_RADIUS).epsilon_at(delta)
        eps_ref = _ref_gaussian_epsilon(sigma, delta)

        assert eps_trunc == pytest.approx(eps_ref, abs=0.01), (
            f"TruncatedGaussian(σ={sigma}, R={LARGE_RADIUS}) "
            f"ε@δ={delta}: {eps_trunc:.6f} vs Gaussian ref {eps_ref:.6f}"
        )


# ============================================================================
# 3. Poisson-subsampled convergence — large R vs dp_accounting
# ============================================================================

POISSON_SIGMAS = [0.8, 1.0]
POISSON_RATES = [0.001, 0.01]
POISSON_STEPS = [100, 500]


class TestPoissonTruncatedConvergence:
    """Poisson(TruncatedGaussian(σ, R=50), q) * N ≈ Poisson(Gaussian(σ), q) * N."""

    @pytest.mark.parametrize("sigma", POISSON_SIGMAS)
    @pytest.mark.parametrize("q", POISSON_RATES)
    @pytest.mark.parametrize("steps", POISSON_STEPS)
    def test_poisson_convergence(self, sigma, q, steps):
        eps_trunc = (
            acc.poisson(acc.truncated_gaussian(sigma, LARGE_RADIUS), q) * steps
        ).epsilon_at(1e-5)
        eps_ref = _ref_poisson_gaussian_epsilon(sigma, 1e-5, sample_rate=q, steps=steps)

        assert eps_trunc == pytest.approx(eps_ref, abs=0.01), (
            f"Poisson(TruncGauss(σ={sigma}, R=50), q={q}) * {steps}: "
            f"ε={eps_trunc:.6f} vs ref {eps_ref:.6f}"
        )


# ============================================================================
# 4. Sanity checks — invariants that any correct implementation must satisfy
# ============================================================================


class TestTruncatedGaussianInvariants:
    """Basic invariants for the truncated Gaussian mechanism."""

    @pytest.mark.parametrize("sigma", [0.3, 0.5, 0.8, 1.0])
    @pytest.mark.parametrize("radius", [2.0, 3.0, 5.0])
    def test_advantage_equals_delta_at_zero(self, sigma, radius):
        """advantage() == delta_at(0)."""
        proc = acc.truncated_gaussian(sigma, radius)
        adv = proc.advantage()
        d0 = proc.delta_at(0.0)
        assert adv == pytest.approx(d0, abs=1e-8), (
            f"TruncatedGaussian(σ={sigma}, R={radius}): "
            f"advantage={adv} != delta_at(0)={d0}"
        )

    @pytest.mark.parametrize("sigma", [0.5, 0.8, 1.0])
    @pytest.mark.parametrize("radius", [2.0, 3.0, 5.0])
    def test_epsilon_delta_roundtrip(self, sigma, radius):
        """ε = epsilon_at(δ) → delta_at(ε) ≈ δ."""
        proc = acc.truncated_gaussian(sigma, radius)
        delta = 1e-5
        eps = proc.epsilon_at(delta)
        delta_back = proc.delta_at(eps)
        assert delta_back == pytest.approx(delta, abs=1e-6), (
            f"TruncatedGaussian(σ={sigma}, R={radius}): "
            f"δ={delta} → ε={eps} → δ'={delta_back}"
        )

    @pytest.mark.parametrize("sigma", [0.5, 0.8, 1.0])
    def test_radius_monotonicity(self, sigma):
        """Larger R → higher ε (closer to unbounded Gaussian)."""
        radii = [1.0, 2.0, 3.0, 5.0, 10.0]
        epsilons = [acc.truncated_gaussian(sigma, r).epsilon_at(1e-5) for r in radii]
        for i in range(len(epsilons) - 1):
            assert epsilons[i] <= epsilons[i + 1] + 1e-6, (
                f"truncated_gaussian(σ={sigma}): "
                f"ε(R={radii[i]})={epsilons[i]:.6f} > "
                f"ε(R={radii[i + 1]})={epsilons[i + 1]:.6f}"
            )

    @pytest.mark.parametrize("radius", [2.0, 5.0])
    def test_noise_monotonicity(self, radius):
        """Higher σ → lower ε (more noise = more private)."""
        sigmas = [0.3, 0.5, 0.8, 1.0]
        epsilons = [acc.truncated_gaussian(s, radius).epsilon_at(1e-5) for s in sigmas]
        for i in range(len(epsilons) - 1):
            assert epsilons[i] >= epsilons[i + 1] - 1e-6, (
                f"truncated_gaussian(R={radius}): "
                f"ε(σ={sigmas[i]})={epsilons[i]:.6f} < "
                f"ε(σ={sigmas[i + 1]})={epsilons[i + 1]:.6f}"
            )
