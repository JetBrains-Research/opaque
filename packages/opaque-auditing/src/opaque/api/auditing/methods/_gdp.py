"""μ-GDP order-statistics audit method.

Mirrors :class:`opaque.accounting.Pld`'s metric surface: ``epsilon_at``,
``delta_at``, ``beta_at``, ``advantage``.  All four derive from a single
inferred μ̂ via :meth:`GdpMethod._mu_at`.  Constructed via
:meth:`OneRunEstimate.gdp`.

Reference: Xiang, Chen, Kerkouche (2025), https://arxiv.org/abs/2509.08704
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import scipy.stats

from opaque.api.auditing.methods import _gdp_math
from opaque.api.auditing.one_run._stats import (
    search_ceiling,
    validate_delta,
    validate_significance,
)

if TYPE_CHECKING:
    from opaque.api.auditing.one_run._estimate import OneRunEstimate


_TOL_MU = 0.01


@dataclasses.dataclass(frozen=True)
class GdpMethod:
    """μ-GDP order-statistics audit method (Xiang et al. 2025)."""

    _estimate: OneRunEstimate
    grid_size: int = 10_000

    # ------------------------------------------------------------------
    # Primitive — inferred μ̂ from the order-statistics test
    # ------------------------------------------------------------------

    def _mu_at(self, significance: float, threshold: float | None) -> float:
        """Inferred μ̂ via binary search.  Independent of δ."""
        validate_significance(significance)
        m = self._estimate.n_in + self._estimate.n_out
        r, u = self._estimate._best_r_u(threshold)
        mu_hi = search_ceiling(m, 0.0, significance)

        mu_lo = 0.0
        while mu_hi - mu_lo > _TOL_MU:
            mu_mid = (mu_lo + mu_hi) / 2.0
            if _gdp_math.p_value(m, r, u, mu_mid, self.grid_size) < significance:
                mu_lo = mu_mid
            else:
                mu_hi = mu_mid
        return mu_lo

    # ------------------------------------------------------------------
    # Pld-mirror surface
    # ------------------------------------------------------------------

    def epsilon_at(
        self,
        *,
        delta: float,
        significance: float = 0.05,
        threshold: float | None = None,
    ) -> float:
        """Epsilon lower bound at the given (δ, significance).

        Raises:
            ValueError: If ``delta <= 0``.
        """
        if delta <= 0:
            raise ValueError(
                f"μ-GDP f-DP auditing requires delta > 0, got {delta}"
            )
        validate_delta(delta)
        return _gdp_math.gdp_to_eps_delta(
            self._mu_at(significance, threshold), delta,
        )

    def delta_at(
        self,
        *,
        epsilon: float,
        significance: float = 0.05,
        threshold: float | None = None,
    ) -> float:
        """δ(ε) under the inferred μ̂-GDP guarantee.

        Closed form: δ(ε; μ) = Φ(μ/2 − ε/μ) − e^ε · Φ(−μ/2 − ε/μ).
        """
        if epsilon < 0:
            raise ValueError(f"epsilon must be >= 0, got {epsilon}")
        mu = self._mu_at(significance, threshold)
        if mu == 0.0:
            return 0.0
        a = mu / 2.0 - epsilon / mu
        b = -mu / 2.0 - epsilon / mu
        term1 = scipy.stats.norm.cdf(a)
        log_term2 = epsilon + scipy.stats.norm.logcdf(b)
        term2 = math.exp(log_term2) if log_term2 < 700 else math.inf
        return float(max(0.0, term1 - term2))

    def beta_at(
        self,
        *,
        alpha: float,
        significance: float = 0.05,
        threshold: float | None = None,
    ) -> float:
        """f-DP Type-II error at α under the inferred μ̂-GDP.

        β(α; μ) = Φ(Φ⁻¹(1 − α) − μ).  Note: this is the *theoretical* β
        of the post-audit guarantee, distinct from
        :meth:`OneRunEstimate.beta_at` which is the empirical attack ROC.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        mu = self._mu_at(significance, threshold)
        return float(
            scipy.stats.norm.cdf(scipy.stats.norm.ppf(1.0 - alpha) - mu)
        )

    def advantage(
        self,
        *,
        significance: float = 0.05,
        threshold: float | None = None,
    ) -> float:
        """Total-variation advantage at the inferred μ̂-GDP.

        TV(μ) = 2 · Φ(μ/2) − 1.
        """
        mu = self._mu_at(significance, threshold)
        return float(2.0 * scipy.stats.norm.cdf(mu / 2.0) - 1.0)
