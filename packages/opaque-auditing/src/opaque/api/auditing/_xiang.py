"""Xiang et al. (2025) order-statistics p-values for privacy auditing.

Two p-value functions using different f-DP hypothesis families:

- :func:`xiang_p_value_eps_delta` — analytical Binomial p-value for
  (ε, δ)-DP hypotheses.  O(1), mechanism-agnostic.
- :func:`xiang_p_value_gaussian` — grid-based order-statistics + Chernoff
  bound for Gaussian (μ-GDP) hypotheses.  O(r' × grid), tightest for DP-SGD.

Reference: Xiang, Chen & Kerkouche, "Tight Privacy Auditing in One Run",
arXiv 2509.08704, 2025.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.special
import scipy.stats

from opaque.api.auditing._gaussian_trade_off import BaseGrid, gaussian_base_pair_grid

__all__ = ["xiang_p_value_eps_delta", "xiang_p_value_gaussian"]

# Maximum number of ranks to compute exactly in the Gaussian path.
# Higher ranks (sorted by |L|) have lower error probability (more confident).
# Truncated low-confidence ranks use v_k = 0.5 (conservative).
# 2000 ranks × 10K grid ≈ 160 MB; covers the top 20% for n = 10K, which
# dominates the Chernoff sum.  Increasing beyond 2000 gives diminishing
# returns since low-confidence ranks contribute v_k ≈ 0.5 anyway.
_MAX_EXACT_RANKS = 2000


# ---------------------------------------------------------------------------
# (ε, δ)-DP path — analytical
# ---------------------------------------------------------------------------


def xiang_p_value_eps_delta(
    r: int,
    u: int,
    eps: float,
    delta: float,
) -> float:
    """P-value under (ε, δ)-DP using Xiang's order-statistics framework.

    For the (ε, δ)-DP trade-off function, all ranks in the continuous
    region have |L| = ε, giving v_k = sigmoid(−ε) for n_eff = r·(1−δ)
    effective ranks.  The remaining r·δ ranks sit in the point-mass
    region (|L| = ∞, v_k = 0).  The Chernoff bound collapses to an
    exact Binomial CDF.

    Args:
        r: Number of released guesses (typically r = m = total canaries).
        u: Number of errors among released guesses.
        eps: Privacy parameter ε to test.
        delta: Privacy parameter δ.

    Returns:
        Upper bound on P(≤ u errors | mechanism is (ε, δ)-DP).
    """
    p = 0.5 if eps <= 0.0 else scipy.special.expit(-eps)
    n_eff = max(r - round(r * delta), 0)

    if n_eff == 0:
        return 1.0

    return float(scipy.stats.binom.cdf(u, n_eff, p))


# ---------------------------------------------------------------------------
# Gaussian (μ-GDP) path — grid-based order statistics + Chernoff
# ---------------------------------------------------------------------------


def xiang_p_value_gaussian(
    n: int,
    r: int,
    u: int,
    mu: float,
    grid_size: int = 10_000,
) -> float:
    """P-value under μ-GDP using Xiang's full order-statistics bound.

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

    # ---- Step 1: Build grid ----
    grid = gaussian_base_pair_grid(mu, grid_size)

    # ---- Step 2: Compute v_k for top r' ranks ----
    r_prime = min(r, _MAX_EXACT_RANKS)
    v_k = _compute_v_k(n, r_prime, grid)

    # ---- Step 3: Chernoff bound ----
    n_trunc = r - r_prime  # truncated ranks use v_k = 0.5
    return _chernoff_lower_tail(v_k, n_trunc, u)


def _compute_v_k(
    n: int,
    r_prime: int,
    grid: BaseGrid,
) -> np.ndarray:
    """Compute v_k for ranks k = n − r' + 1, …, n.

    v_k = E[sigmoid(−|L(Y_{(k)})|)] where Y_{(k)} is the k-th order
    statistic of n i.i.d. draws from the mixture density f_Y
    (Xiang et al. 2025, Eq. 12).

    Returns array of shape (r_prime,) with v_k values in [0, 0.5].
    """
    k_vals = np.arange(n - r_prime + 1, n + 1, dtype=np.float64)

    # Log combinatorial: ln(n! / ((k-1)! · (n-k)!))
    log_C = (
        scipy.special.gammaln(n + 1)
        - scipy.special.gammaln(k_vals)
        - scipy.special.gammaln(n - k_vals + 1)
    )

    # Grid quantities — all shape (grid_size,)
    mass = grid.mass
    F = grid.F_y
    abs_pl = grid.abs_privacy_loss

    log_mass = np.log(np.maximum(mass, 1e-300))
    log_F = np.log(np.maximum(F, 1e-300))
    log_1mF = np.log(np.maximum(1.0 - F, 1e-300))
    log_sig = scipy.special.log_expit(-abs_pl)

    # Broadcast: k is (r_prime, 1), grid is (1, grid_size)
    k = k_vals[:, None]

    # (k-1) · log(F_Y):  0 when k = 1 to avoid 0 · (-inf) = NaN
    term_F = np.where(k > 1, (k - 1) * log_F[None, :], 0.0)

    # (n-k) · log(1 - F_Y):  0 when k = n
    term_1mF = np.where(k < n, (n - k) * log_1mF[None, :], 0.0)

    # Full log integrand: (r_prime, grid_size)
    log_integrand = (
        log_C[:, None] + log_mass[None, :] + term_F + term_1mF + log_sig[None, :]
    )

    # v_k = Σ_grid exp(log_integrand)  —  logsumexp for stability
    log_v_k = scipy.special.logsumexp(log_integrand, axis=1)

    # v_k ∈ [0, 0.5] since sigmoid(−|L|) ≤ 0.5
    return np.clip(np.exp(log_v_k), 0.0, 0.5)


def _chernoff_lower_tail(
    v_k: np.ndarray,
    n_trunc: int,
    u: int,
) -> float:
    """Chernoff bound on P(Σ V_k ≤ u)  (Xiang et al. 2025, Theorem 3).

    V_k are independent Bernoulli(v_k) for the exactly-computed ranks,
    plus n_trunc independent Bernoulli(0.5) for the truncated ranks.

    Returns the minimised Chernoff upper bound on the lower tail, or
    1.0 if the observed errors u are not below expectation.
    """
    # Expected total errors
    expected = float(np.sum(v_k)) + n_trunc * 0.5

    if u >= expected:
        return 1.0  # adversary didn't beat expectation — can't reject

    if u == 0:
        # All guesses correct.  The Chernoff minimum is at λ → −∞,
        # giving P = Π(1−v_k) · 0.5^n_trunc.
        log_pval = float(
            np.sum(np.log(np.maximum(1.0 - v_k, 1e-300)))
        ) + n_trunc * math.log(0.5)
        return min(math.exp(log_pval), 1.0)

    # Bisect for λ* ∈ (−50, 0) where dκ/dλ = 0.
    # κ(λ) = −λu + Σ ln(1 − v_k + v_k·e^λ) + n_trunc·ln((1+e^λ)/2)
    # dκ/dλ = −u + Σ v_k·e^λ/(1−v_k+v_k·e^λ) + n_trunc·sigmoid(λ)
    def _kappa_deriv(lam: float) -> float:
        e_lam = math.exp(lam)
        exact = float(np.sum(v_k * e_lam / (1.0 - v_k + v_k * e_lam)))
        trunc = n_trunc * e_lam / (1.0 + e_lam)
        return -u + exact + trunc

    lam_lo, lam_hi = -50.0, 0.0

    # Verify bracket: deriv should be < 0 at lam_lo, > 0 at lam_hi
    if _kappa_deriv(lam_lo) >= 0:
        # λ* is more negative than -50; evaluate κ at lam_lo (conservative)
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

    # Evaluate κ(λ*)
    e_lam = math.exp(lam_star)
    kappa = (
        -lam_star * u
        + float(np.sum(np.log(1.0 - v_k + v_k * e_lam)))
        + n_trunc * math.log((1.0 + e_lam) / 2.0)
    )

    return min(math.exp(kappa), 1.0)
