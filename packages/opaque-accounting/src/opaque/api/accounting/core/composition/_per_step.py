"""Per-step algebraic view of a whole-horizon DP process."""

from __future__ import annotations

from dataclasses import dataclass

from opaque.api.accounting.core._base import DpProcess, Pld
from opaque.api.accounting.core._horizon import DpHorizonProcess
from opaque.api.accounting.core._pld_cache import pld_cache
from opaque.exceptions import ConfigurationError, InputTypeError


@dataclass(frozen=True, slots=True)
class PerStep(DpProcess):
    """Expose a :class:`DpHorizonProcess` through ordinary step composition."""

    process: DpHorizonProcess

    def __post_init__(self) -> None:
        if not isinstance(self.process, DpHorizonProcess):
            raise InputTypeError(
                *(
                    f"PerStep requires a DpHorizonProcess, got "
                    f"{type(self.process).__name__}.",
                )
            )

    def _pld_cache_key(self, *, n_steps: int | None = None) -> tuple[object, ...]:
        return (
            "PerStep",
            self.process._pld_cache_key(n_steps=1 if n_steps is None else n_steps),
        )

    @pld_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        max_conv_grid: int | None = None,
        seed: int | None = None,
        mc_resolution: float | None = None,
        mc_failure_probability: float | None = None,
    ) -> Pld:
        return self.process.pld_at(
            1,
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            max_conv_grid=max_conv_grid,
            seed=seed,
            mc_resolution=mc_resolution,
            mc_failure_probability=mc_failure_probability,
        )

    def repeated_pld(
        self,
        count: int,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        max_conv_grid: int | None = None,
        seed: int | None = None,
        mc_resolution: float | None = None,
        mc_failure_probability: float | None = None,
    ) -> Pld:
        if count < 1:
            raise ConfigurationError(*(f"count ({count}) must be >= 1",))
        if count > self.process.n_steps:
            raise ConfigurationError(
                *(
                    f"count ({count}) exceeds n_steps ({self.process.n_steps}); "
                    f"{type(self.process).__name__} is undefined beyond its "
                    "declared horizon.",
                )
            )
        return self.process.pld_at(
            count,
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            max_conv_grid=max_conv_grid,
            seed=seed,
            mc_resolution=mc_resolution,
            mc_failure_probability=mc_failure_probability,
        )


def per_step(process: DpHorizonProcess) -> PerStep:
    """Wrap a whole-horizon process for ``acc |= step`` training loops."""
    # The DpHorizonProcess check lives in ``PerStep.__post_init__``.
    return PerStep(process=process)


__all__ = ["PerStep", "per_step"]
