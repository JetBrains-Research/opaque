"""Mechanism-agnostic (ε, δ)-DP order-statistics audit method.

For the (ε, δ)-DP trade-off function, all ranks in the continuous region
have |L| = ε, giving v_k = sigmoid(−ε) for n_eff = r·(1−δ) effective
ranks.  The remaining r·δ ranks sit in the point-mass region
(|L| = ∞, v_k = 0).  The Chernoff bound collapses to an exact Binomial CDF.

Mirrors :class:`opaque.accounting.Pld`'s metric surface: ``epsilon_at``,
``delta_at``, ``beta_at``, ``advantage``.  Every metric is evaluated at
the inferred point ``(ε̂(δ), δ)`` via :meth:`EpsDeltaMethod._epsilon_at`.
Constructed via :meth:`OneRunEstimate.eps_delta`.

Reference: Xiang, Chen, Kerkouche (2025), https://arxiv.org/abs/2509.08704
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import scipy.special
import scipy.stats

from opaque.api.auditing.one_run._stats import (
    search_ceiling,
    validate_delta,
    validate_significance,
)

if TYPE_CHECKING:
    from opaque.api.auditing.one_run._estimate import OneRunEstimate


_TOL = 1e-4
_DELTA_TOL = 1e-6


def _p_value(r: int, u: int, eps: float, delta: float) -> float:
    """P-value under (ε, δ)-DP at ``r`` guesses with ``u`` errors."""
    p = 0.5 if eps <= 0.0 else scipy.special.expit(-eps)
    n_eff = max(r - round(r * delta), 0)
    if n_eff == 0:
        return 1.0
    return float(scipy.stats.binom.cdf(u, n_eff, p))


@dataclasses.dataclass(frozen=True)
class EpsDeltaMethod:
    """(ε, δ)-DP order-statistics audit method (Xiang et al. 2025)."""

    _estimate: OneRunEstimate

    # ------------------------------------------------------------------
    # Primitive — inferred ε̂ at a given δ
    # ------------------------------------------------------------------

    def _epsilon_at(
        self,
        delta: float,
        significance: float,
        threshold: float | None,
    ) -> float:
        validate_significance(significance)
        validate_delta(delta)
        r, u = self._estimate._best_r_u(threshold)
        m = self._estimate.n_in + self._estimate.n_out
        eps_hi = search_ceiling(m, delta, significance)

        eps_lo = 0.0
        while eps_hi - eps_lo > _TOL:
            eps_mid = (eps_lo + eps_hi) / 2.0
            if _p_value(r, u, eps_mid, delta) < significance:
                eps_lo = eps_mid
            else:
                eps_hi = eps_mid
        return eps_lo

    # ------------------------------------------------------------------
    # Pld-mirror surface
    # ------------------------------------------------------------------

    def epsilon_at(
        self,
        *,
        delta: float = 0.0,
        significance: float = 0.05,
        threshold: float | None = None,
    ) -> float:
        """Epsilon lower bound at the given (δ, significance)."""
        return self._epsilon_at(delta, significance, threshold)

    def delta_at(
        self,
        *,
        epsilon: float,
        significance: float = 0.05,
        threshold: float | None = None,
    ) -> float:
        """Largest δ at which the audit certifies ε ≥ ``epsilon``.

        Binary search over δ; relies on monotonicity of the p-value in δ
        at fixed ε.  Returns 0.0 if ``epsilon`` is unreachable even at δ=0.
        """
        validate_significance(significance)
        if epsilon < 0:
            raise ValueError(f"epsilon must be >= 0, got {epsilon}")
        r, u = self._estimate._best_r_u(threshold)

        if _p_value(r, u, epsilon, 0.0) >= significance:
            return 0.0

        delta_lo, delta_hi = 0.0, 1.0
        while delta_hi - delta_lo > _DELTA_TOL:
            delta_mid = (delta_lo + delta_hi) / 2.0
            if _p_value(r, u, epsilon, delta_mid) < significance:
                delta_lo = delta_mid
            else:
                delta_hi = delta_mid
        return delta_lo

    def beta_at(
        self,
        *,
        alpha: float,
        delta: float = 0.0,
        significance: float = 0.05,
        threshold: float | None = None,
    ) -> float:
        """f-DP Type-II error at α under the inferred (ε̂(δ), δ)-DP.

        β(α; ε, δ) = max(0, 1 − δ − e^ε·α, e^(−ε)·(1 − δ − α)).  Note:
        this is the *theoretical* β of the post-audit guarantee, distinct
        from :meth:`OneRunEstimate.beta_at` which is the empirical attack
        ROC.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        eps = self._epsilon_at(delta, significance, threshold)
        return max(
            0.0,
            1.0 - delta - math.exp(eps) * alpha,
            math.exp(-eps) * (1.0 - delta - alpha),
        )

    def advantage(
        self,
        *,
        delta: float = 0.0,
        significance: float = 0.05,
        threshold: float | None = None,
    ) -> float:
        """Total-variation advantage at the inferred (ε̂(δ), δ)-DP.

        TV(ε, δ) = 1 − (1 − δ) · e^(−ε).
        """
        eps = self._epsilon_at(delta, significance, threshold)
        return 1.0 - (1.0 - delta) * math.exp(-eps)
