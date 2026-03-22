"""Homogeneous k-fold self-composition of a DP process."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from opaque_accounting.base import DpProcess, Pld


@dataclass(frozen=True, slots=True)
class Repeated(DpProcess):
    """Homogeneous k-fold self-composition."""

    inner: DpProcess
    count: int

    def _leaf_and_count(self) -> tuple[DpProcess, int]:
        return (self.inner, self.count)

    @functools.lru_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        pessimistic_estimate: bool | None = None,
        max_grid_size: int | None = None,
    ) -> Pld:
        # Try SPA fast-path for small noise multipliers (avoids PLD grid explosion)
        spa = self._try_spa()
        if spa is not None:
            return spa

        return self.inner.pld(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            pessimistic_estimate=pessimistic_estimate,
            max_grid_size=max_grid_size,
        ).self_compose(self.count)

    def _try_spa(self) -> Pld | None:
        """Use Saddle-Point Accountant for small-nm Gaussian/Poisson mechanisms.

        Returns an SPA-backed PLD if applicable, None otherwise.
        The SPA handles small noise multipliers (< 0.1) where PLD discretization
        struggles with grid explosion.
        """
        from opaque_accounting import opaque_accounting as _native
        from opaque_accounting.amplification.poisson import Poisson
        from opaque_accounting.mechanisms.gaussian import Gaussian

        _SPA_THRESHOLD = 0.1

        match self.inner:
            case Gaussian(noise_multiplier=nm) if nm < _SPA_THRESHOLD:
                return _native.spa_gaussian_pld(nm).self_compose(self.count)
            case Poisson(inner=Gaussian(noise_multiplier=nm), sample_rate=q) if nm < _SPA_THRESHOLD:
                return _native.spa_poisson_gaussian_pld(nm, q).self_compose(self.count)
            case _:
                return None
