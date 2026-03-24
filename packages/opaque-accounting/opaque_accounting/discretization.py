"""Private module for PLD discretization configuration management."""

from __future__ import annotations

from dataclasses import dataclass

from . import opaque_accounting as _native


@dataclass(frozen=True, slots=True)
class DiscretizationConfig:
    """Discretization configuration for PLD computation.

    Args:
        discretization: Grid spacing for the PLD PMF. Smaller = tighter, slower.
        log_x_mass_truncation_bound: Log tail mass cutoff in x-space. Tails below exp(bound) are truncated.
        pessimistic_estimate: Round upward for safe upper bounds (True) or downward (False).
        max_grid_size: Maximum grid bins before automatic coarsening.
    """

    discretization: float = 1e-4
    log_x_mass_truncation_bound: float = -50.0
    pessimistic_estimate: bool = True
    max_grid_size: int = 10_000_000

    def to_native(self) -> _native.DiscretizationConfig:
        """Convert to Rust DiscretizationConfig for FFI calls."""
        return _native.DiscretizationConfig(
            discretization=self.discretization,
            log_mass_truncation_bound=self.log_x_mass_truncation_bound,
            pessimistic_estimate=self.pessimistic_estimate,
            max_grid_size=self.max_grid_size,
        )


__all__ = [
    "DiscretizationConfig",
]
