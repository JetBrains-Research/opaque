"""Heterogeneous composition of two DP processes."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from opaque_accounting.base import DpProcess, Pld


@dataclass(frozen=True, slots=True)
class Composed(DpProcess):
    """Heterogeneous composition of two processes."""

    left: DpProcess
    right: DpProcess

    @functools.lru_cache(maxsize=1)
    def cgf(self) -> Pld:
        return self.left.cgf().compose(self.right.cgf())

    @functools.lru_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        pessimistic_estimate: bool | None = None,
        max_grid_size: int | None = None,
    ) -> Pld:
        left_pld = self.left.pld(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            pessimistic_estimate=pessimistic_estimate,
            max_grid_size=max_grid_size,
        )
        right_pld = self.right.pld(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            pessimistic_estimate=pessimistic_estimate,
            max_grid_size=max_grid_size,
        )
        return left_pld.compose(right_pld)
