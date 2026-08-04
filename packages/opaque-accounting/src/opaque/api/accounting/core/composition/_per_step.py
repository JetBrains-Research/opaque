"""Per-step algebraic view of a whole-horizon DP process."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from opaque.api.accounting.core._base import DpProcess, Pld
from opaque.api.accounting.core._horizon import DpHorizonProcess


@dataclass(frozen=True, slots=True)
class PerStep(DpProcess):
    """Expose a :class:`DpHorizonProcess` through ordinary step composition."""

    process: DpHorizonProcess

    @functools.lru_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        max_conv_grid: int | None = None,
    ) -> Pld:
        return self.process.pld_at(
            1,
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            max_conv_grid=max_conv_grid,
        )

    def repeated_pld(
        self,
        count: int,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        max_conv_grid: int | None = None,
    ) -> Pld:
        if count < 1:
            raise ValueError(f"count ({count}) must be >= 1")
        if count > self.process.n_steps:
            raise ValueError(
                f"count ({count}) exceeds n_steps ({self.process.n_steps}); "
                f"{type(self.process).__name__} is undefined beyond its "
                "declared horizon."
            )
        return self.process.pld_at(
            count,
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            max_conv_grid=max_conv_grid,
        )


def per_step(process: DpHorizonProcess) -> PerStep:
    """Wrap a whole-horizon process for ``acc |= step`` training loops."""
    if not isinstance(process, DpHorizonProcess):
        raise TypeError(
            f"per_step() requires a DpHorizonProcess, got {type(process).__name__}."
        )
    return PerStep(process=process)


__all__ = ["PerStep", "per_step"]
