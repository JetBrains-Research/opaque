"""μ-GDP order-statistics audit method.

Wraps the μ-GDP p-value from :mod:`._gdp_math` as a frozen view over a
:class:`OneRunEstimate` plus the grid size used for the numerical
integration.  Constructed via :meth:`OneRunEstimate.gdp`.

Reference: Xiang, Chen, Kerkouche (2025), https://arxiv.org/abs/2509.08704
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

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

    def epsilon_at(
        self,
        *,
        delta: float,
        significance: float = 0.05,
        threshold: float | None = None,
    ) -> float:
        """Epsilon lower bound at the given (δ, significance).

        Args:
            delta: DP delta parameter. Must be > 0 — Gaussian DP cannot
                satisfy pure DP.
            significance: Allowed failure probability (1 − confidence).
            threshold: If provided, score threshold to test at. Otherwise the
                Pareto-optimal threshold maximising TP + TN is used.

        Raises:
            ValueError: If ``delta <= 0``.
        """
        if delta <= 0:
            raise ValueError(
                f"Gaussian f-DP auditing requires delta > 0, got {delta}"
            )
        validate_delta(delta)
        validate_significance(significance)

        m = self._estimate.n_in + self._estimate.n_out
        r, u = self._estimate._best_r_u(threshold)
        mu_hi = search_ceiling(m, delta, significance)

        mu_lo = 0.0
        while mu_hi - mu_lo > _TOL_MU:
            mu_mid = (mu_lo + mu_hi) / 2.0
            if _gdp_math.p_value(m, r, u, mu_mid, self.grid_size) < significance:
                mu_lo = mu_mid
            else:
                mu_hi = mu_mid

        return _gdp_math.gdp_to_eps_delta(mu_lo, delta)
