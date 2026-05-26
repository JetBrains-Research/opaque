"""Numerical machinery for the μ-GDP one-run audit method.

Contains:

- :func:`gdp_to_eps_delta` — closed-form μ-GDP → (ε, δ) conversion.
- :func:`gdp_base_pair_grid` + :class:`BaseGrid` — discretised base
  distribution pair used by the order-statistics p-value.
- :func:`p_value` — μ-GDP p-value via order statistics + Chernoff bound
  (Xiang et al. 2025, Theorems 2–3).

Torch-free: numpy + scipy only.

Reference: Xiang, Chen, Kerkouche (2025), https://arxiv.org/abs/2509.08704
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import scipy.special
import scipy.stats

# Maximum number of ranks to compute exactly.  Higher ranks (sorted by |L|)
# have lower error probability (more confident).  Truncated low-confidence
# ranks use v_k = 0.5 (conservative).  2000 ranks × 10K grid ≈ 160 MB;
# covers the top 20% for n = 10K, which dominates the Chernoff sum.
# Increasing beyond 2000 gives diminishing returns since low-confidence
# ranks contribute v_k ≈ 0.5 anyway.
_MAX_EXACT_RANKS = 2000


# ---------------------------------------------------------------------------
# GDP → (ε, δ) conversion
# ---------------------------------------------------------------------------


def gdp_to_eps_delta(mu: float, delta: float) -> float:
    """Convert μ-GDP to (ε, δ)-DP.  Returns ε.

    Uses the closed-form relation
        δ(ε) = Φ(μ/2 − ε/μ) − exp(ε)·Φ(−μ/2 − ε/μ)
    and binary-searches for the ε at which δ(ε) = *delta*.

    Args:
        mu: Gaussian DP parameter (σ = 1/μ for sensitivity-1 queries).
            Must be ≥ 0.
        delta: Target failure probability.  Must be in (0, 1).

    Returns:
        Smallest ε such that the μ-GDP mechanism satisfies (ε, δ)-DP.

    Raises:
        ValueError: If mu < 0 or delta is not in (0, 1).
    """
    if mu < 0.0:
        raise ValueError(f"mu must be >= 0, got {mu}")
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if mu == 0.0:
        return 0.0

    _norm_cdf = scipy.stats.norm.cdf
    _norm_logcdf = scipy.stats.norm.logcdf

    def _delta_at_eps(eps: float) -> float:
        a = mu / 2.0 - eps / mu
        b = -mu / 2.0 - eps / mu
        term1 = _norm_cdf(a)
        # exp(eps) * Phi(b) in log-space to avoid overflow for large eps.
        log_term2 = eps + _norm_logcdf(b)
        term2 = math.exp(log_term2) if log_term2 < 700 else math.inf
        return term1 - term2

    eps_lo = 0.0
    eps_hi = mu * mu + 4.0 * mu
    while _delta_at_eps(eps_hi) > delta:
        eps_hi *= 2.0

    if _delta_at_eps(0.0) <= delta:
        return 0.0

    for _ in range(100):
        eps_mid = (eps_lo + eps_hi) / 2.0
        if _delta_at_eps(eps_mid) > delta:
            eps_lo = eps_mid
        else:
            eps_hi = eps_mid
        if eps_hi - eps_lo < 1e-12:
            break

    return eps_lo


# ---------------------------------------------------------------------------
# Discretised base distribution pair for μ-GDP
# ---------------------------------------------------------------------------


class BaseGrid(NamedTuple):
    """Discretised base distribution pair for μ-GDP.

    All arrays are sorted by ascending ``abs_privacy_loss``.  The grid
    is constructed in z-space (z = Φ⁻¹(y)) for numerical stability at
    all μ values.  ``F_y`` is the CDF of the mixture density in
    |L|-sorted space, regardless of the z-space parameterisation.
    """

    z: np.ndarray
    """z-space grid points (sorted by |L|, not by z-value)."""

    mass: np.ndarray
    """Mass per grid point: (φ(z) + φ(z−μ)) / 2 · Δz."""

    F_y: np.ndarray
    """CDF of the mixture density in |L|-sorted space."""

    abs_privacy_loss: np.ndarray
    """|L(z)| = |μ²/2 − μ·z| at each grid point."""


def gdp_base_pair_grid(mu: float, num_points: int) -> BaseGrid:
    """Build discretised base pair for μ-GDP.

    The grid lives in z-space where z = Φ⁻¹(y).  In this space:

    - P density in z:  φ(z)      (standard normal)
    - Q density in z:  φ(z − μ)  (shifted normal)
    - Mixture:  (φ(z) + φ(z − μ)) / 2
    - |L(z)| = |μ²/2 − μ·z|

    Both densities are smooth Gaussians, so a uniform z-grid captures
    all the mass regardless of μ.

    Args:
        mu: Gaussian DP parameter (must be > 0).
        num_points: Number of grid points.

    Returns:
        A :class:`BaseGrid` with arrays sorted by ascending |L|.
    """
    if mu <= 0.0:
        raise ValueError("mu must be > 0 for grid construction")

    z_lo = -6.0
    z_hi = mu + 6.0
    z = np.linspace(z_lo, z_hi, num_points)
    dz = z[1] - z[0]

    phi_z = scipy.stats.norm.pdf(z)
    phi_z_mu = scipy.stats.norm.pdf(z - mu)

    mix_dz = (phi_z + phi_z_mu) / 2.0
    mass = mix_dz * dz

    # L(z) = ln(φ(z)/φ(z−μ)) = −μ·z + μ²/2  ⇒  |L(z)| = |μ²/2 − μ·z|
    abs_pl = np.abs(0.5 * mu * mu - mu * z)

    sort_idx = np.argsort(abs_pl, kind="stable")
    z_sorted = z[sort_idx]
    mass_sorted = mass[sort_idx]
    abs_pl_sorted = abs_pl[sort_idx]

    cum = np.cumsum(mass_sorted)
    F_y = np.empty_like(cum)
    F_y[0] = 0.0
    F_y[1:] = cum[:-1]

    return BaseGrid(
        z=z_sorted,
        mass=mass_sorted,
        F_y=F_y,
        abs_privacy_loss=abs_pl_sorted,
    )


# ---------------------------------------------------------------------------
# Order-statistics + Chernoff p-value
# ---------------------------------------------------------------------------


def p_value(
    n: int,
    r: int,
    u: int,
    mu: float,
    grid_size: int = 10_000,
) -> float:
    """P-value under μ-GDP.

    Builds a discretised base pair for the Gaussian trade-off function,
    computes the conditional error probability v_k for the top r' ranks
    via numerical integration (Theorem 2, Eq. 12 of Xiang et al. 2025),
    then applies a Chernoff tail bound on the sum of independent
    heterogeneous Bernoullis (Theorem 3).

    Args:
        n: Total number of canaries.
        r: Number of released guesses (typically r = n).
        u: Number of errors among released guesses.
        mu: Gaussian DP parameter to test (σ = 1/μ).
        grid_size: Number of grid points for numerical integration.

    Returns:
        Upper bound on P(≤ u errors | mechanism is μ-GDP).
    """
    if mu <= 0.0:
        return 1.0  # perfectly private — can't reject

    grid = gdp_base_pair_grid(mu, grid_size)
    r_prime = min(r, _MAX_EXACT_RANKS)
    v_k = _compute_v_k(n, r_prime, grid)
    n_trunc = r - r_prime  # truncated ranks use v_k = 0.5
    return _chernoff_lower_tail(v_k, n_trunc, u)


def _compute_v_k(n: int, r_prime: int, grid: BaseGrid) -> np.ndarray:
    """v_k for ranks k = n − r' + 1, …, n.

    v_k = E[sigmoid(−|L(Y_{(k)})|)] where Y_{(k)} is the k-th order
    statistic of n i.i.d. draws from the mixture density f_Y
    (Xiang et al. 2025, Eq. 12).  Returns shape ``(r_prime,)`` in [0, 0.5].
    """
    k_vals = np.arange(n - r_prime + 1, n + 1, dtype=np.float64)

    log_C = (
        scipy.special.gammaln(n + 1)
        - scipy.special.gammaln(k_vals)
        - scipy.special.gammaln(n - k_vals + 1)
    )

    mass = grid.mass
    F = grid.F_y
    abs_pl = grid.abs_privacy_loss

    log_mass = np.log(np.maximum(mass, 1e-300))
    log_F = np.log(np.maximum(F, 1e-300))
    log_1mF = np.log(np.maximum(1.0 - F, 1e-300))
    log_sig = scipy.special.log_expit(-abs_pl)

    k = k_vals[:, None]

    # (k-1) · log(F_Y):  0 when k = 1 to avoid 0 · (-inf) = NaN
    term_F = np.where(k > 1, (k - 1) * log_F[None, :], 0.0)
    # (n-k) · log(1 - F_Y):  0 when k = n
    term_1mF = np.where(k < n, (n - k) * log_1mF[None, :], 0.0)

    log_integrand = (
        log_C[:, None] + log_mass[None, :] + term_F + term_1mF + log_sig[None, :]
    )
    log_v_k = scipy.special.logsumexp(log_integrand, axis=1)

    return np.clip(np.exp(log_v_k), 0.0, 0.5)


def _chernoff_lower_tail(v_k: np.ndarray, n_trunc: int, u: int) -> float:
    """Chernoff bound on P(Σ V_k ≤ u)  (Xiang et al. 2025, Theorem 3).

    V_k are independent Bernoulli(v_k) for the exactly-computed ranks,
    plus n_trunc independent Bernoulli(0.5) for the truncated ranks.
    Returns the minimised upper bound, or 1.0 if observed errors are not
    below expectation.
    """
    expected = float(np.sum(v_k)) + n_trunc * 0.5
    if u >= expected:
        return 1.0

    if u == 0:
        # λ → −∞ minimum: P = Π(1−v_k) · 0.5^n_trunc
        log_pval = float(
            np.sum(np.log(np.maximum(1.0 - v_k, 1e-300)))
        ) + n_trunc * math.log(0.5)
        return min(math.exp(log_pval), 1.0)

    # Bisect for λ* ∈ (−50, 0) where dκ/dλ = 0.
    # κ(λ) = −λu + Σ ln(1 − v_k + v_k·e^λ) + n_trunc·ln((1+e^λ)/2)
    def _kappa_deriv(lam: float) -> float:
        e_lam = math.exp(lam)
        exact = float(np.sum(v_k * e_lam / (1.0 - v_k + v_k * e_lam)))
        trunc = n_trunc * e_lam / (1.0 + e_lam)
        return -u + exact + trunc

    lam_lo, lam_hi = -50.0, 0.0
    if _kappa_deriv(lam_lo) >= 0:
        lam_star = lam_lo
    else:
        for _ in range(100):
            lam_mid = (lam_lo + lam_hi) / 2.0
            if _kappa_deriv(lam_mid) > 0:
                lam_hi = lam_mid
            else:
                lam_lo = lam_mid
            if lam_hi - lam_lo < 1e-10:
                break
        lam_star = (lam_lo + lam_hi) / 2.0

    e_lam = math.exp(lam_star)
    kappa = (
        -lam_star * u
        + float(np.sum(np.log(1.0 - v_k + v_k * e_lam)))
        + n_trunc * math.log((1.0 + e_lam) / 2.0)
    )
    return min(math.exp(kappa), 1.0)
