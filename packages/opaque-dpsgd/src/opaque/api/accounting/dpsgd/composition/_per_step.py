"""Per-step view of a whole-epoch random-allocation accountant."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from opaque.api.accounting.core._base import DpProcess, Pld
from opaque.api.accounting.dpsgd.amplification._random_allocation import (
    RandomAllocation,
)


@dataclass(frozen=True, slots=True)
class PerStepRandomAllocation(DpProcess):
    """Expose a random-allocation epoch process through the per-step algebra.

    ``RandomAllocation`` is exact at epoch boundaries. For a partial epoch,
    this adapter conservatively charges the entire containing epoch, so
    ``step * K`` remains a valid privacy bound for every ``K`` up to
    ``n_steps``.
    """

    process: RandomAllocation
    n_steps: int

    def __post_init__(self) -> None:
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {self.n_steps}")

    @functools.lru_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
    ) -> Pld:
        return self.repeated_pld(
            1,
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
        )

    def repeated_pld(
        self,
        count: int,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
    ) -> Pld:
        if count < 1:
            raise ValueError(f"count ({count}) must be >= 1")
        if count > self.n_steps:
            raise ValueError(
                f"count ({count}) exceeds n_steps ({self.n_steps}); "
                "random-allocation per-step accounting is undefined beyond "
                "its declared horizon."
            )
        epochs = -(-count // self.process.steps_per_epoch)
        return self.process.repeated_pld(
            epochs,
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
        )

    def __or__(self, other: DpProcess) -> DpProcess:
        if isinstance(other, PerStepRandomAllocation) and other != self:
            raise ValueError(
                "PerStepRandomAllocation cannot be composed with a different "
                "random-allocation process or horizon."
            )
        return DpProcess.__or__(self, other)


def per_step(process: RandomAllocation, *, n_steps: int) -> PerStepRandomAllocation:
    """Expose a random-allocation epoch accountant as a composable step view.

    Partial epochs are charged as full epochs, so this remains conservative
    while allowing ordinary ``acc |= step`` training loops.
    """

    if not isinstance(process, RandomAllocation):
        raise TypeError(
            "per_step() requires a RandomAllocation process, got "
            f"{type(process).__name__}."
        )
    return PerStepRandomAllocation(process=process, n_steps=n_steps)


__all__ = ["PerStepRandomAllocation", "per_step"]
