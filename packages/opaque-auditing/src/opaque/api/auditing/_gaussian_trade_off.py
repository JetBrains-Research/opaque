"""Gaussian DP closed-form helpers for Xiang et al. (2025) auditing.

Provides the GDP-to-(ε,δ) conversion and the discretised base distribution
pair needed by the order-statistics p-value (``xiang_p_value_gaussian``).
Torch-free: numpy + scipy only.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import scipy.stats

__all__ = ["BaseGrid", "gaussian_base_pair_grid", "gaussian_to_eps_delta"]


# ---------------------------------------------------------------------------
# GDP → (ε, δ) conversion
# ---------------------------------------------------------------------------

def gaussian_to_eps_delta(mu: float, delta: float) -> float:
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
        # math.exp overflows around 709, so short-circuit above that.
        log_term2 = eps + _norm_logcdf(b)
        term2 = math.exp(log_term2) if log_term2 < 700 else math.inf
        return term1 - term2

    # Upper bound: for the Gaussian mechanism, ε ≤ μ²/2 + μ√(2 ln(1/δ)).
    # The simpler μ² + 4μ is a generous initial bracket that covers
    # all practical (μ, δ) pairs; auto-expanded below if needed.
    eps_lo = 0.0
    eps_hi = mu * mu + 4.0 * mu

    # Expand until δ(eps_hi) < delta (guaranteed since δ → 0 as ε → ∞)
    while _delta_at_eps(eps_hi) > delta:
        eps_hi *= 2.0

    # δ(0) might already be ≤ delta for very large δ
    if _delta_at_eps(0.0) <= delta:
        return 0.0

    # Binary search: ~60 iterations gives 1e-18 precision, 100 is generous
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


def gaussian_base_pair_grid(mu: float, num_points: int) -> BaseGrid:
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

    # z-grid covering both N(0,1) and N(μ,1) within 6 standard deviations
    z_lo = -6.0
    z_hi = mu + 6.0
    z = np.linspace(z_lo, z_hi, num_points)
    dz = z[1] - z[0]

    # Densities in z-space
    phi_z = scipy.stats.norm.pdf(z)          # P in z-space
    phi_z_mu = scipy.stats.norm.pdf(z - mu)  # Q in z-space

    # Mixture density contribution per grid point
    mix_dz = (phi_z + phi_z_mu) / 2.0  # (P + Q) / 2 in z-space
    mass = mix_dz * dz

    # Privacy loss: L(z) = ln(φ(z)/φ(z−μ)) = −μ·z + μ²/2
    # |L(z)| = |μ²/2 − μ·z|
    abs_pl = np.abs(0.5 * mu * mu - mu * z)

    # Sort by ascending |L|
    sort_idx = np.argsort(abs_pl, kind="stable")
    z_sorted = z[sort_idx]
    mass_sorted = mass[sort_idx]
    abs_pl_sorted = abs_pl[sort_idx]

    # CDF: cumulative mass in |L|-sorted order
    # F_y[i] = mass of all points with index < i (strictly)
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
