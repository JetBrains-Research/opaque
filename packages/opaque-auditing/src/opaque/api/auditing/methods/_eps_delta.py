"""Mechanism-agnostic (ε, δ)-DP order-statistics audit method.

For the (ε, δ)-DP trade-off function, all ranks in the continuous region
have |L| = ε, giving v_k = sigmoid(−ε) for n_eff = r·(1−δ) effective
ranks.  The remaining r·δ ranks sit in the point-mass region
(|L| = ∞, v_k = 0).  The Chernoff bound collapses to an exact Binomial CDF.

Reference: Xiang, Chen, Kerkouche (2025), https://arxiv.org/abs/2509.08704
"""

from __future__ import annotations

import dataclasses
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

    def epsilon_at(
        self,
        *,
        delta: float = 0.0,
        significance: float = 0.05,
        threshold: float | None = None,
    ) -> float:
        """Epsilon lower bound at the given (δ, significance).

        Args:
            delta: DP delta parameter.
            significance: Allowed failure probability (1 − confidence).
            threshold: If provided, score threshold to test at. Otherwise the
                Pareto-optimal threshold maximising TP + TN is used.
        """
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
