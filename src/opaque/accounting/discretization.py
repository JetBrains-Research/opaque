"""Private module for PLD discretization configuration management."""

from __future__ import annotations

from dataclasses import dataclass

import opaque_accounting as _native


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
    "set_discretization",
    "get_discretization",
]

# Module-level discretization default
_default_config: DiscretizationConfig | None = None


def set_discretization(
    discretization: float = 1e-4,
    log_x_mass_truncation_bound: float = -50.0,
    pessimistic_estimate: bool = True,
    max_grid_size: int = 10_000_000,
) -> None:
    """Set module-level default discretization parameters.

    These defaults are used when query parameters are not provided.
    By default, uses high-precision settings matching Google's dp_accounting.

    Args:
        discretization: Grid spacing for PLD PMF. Smaller = more precise, larger = faster.
            Error scales as O(disc^2). Default: 1e-4.
        log_x_mass_truncation_bound: Tails with log-probability below this bound in x-space
            are truncated. Default: -50 (matching Google).
        pessimistic_estimate: If True (default), round probabilities upward to
            produce an **upper bound** on privacy loss (safe for guarantees). If
            False, round downward (optimistic estimate, useful for debugging only).
        max_grid_size: If grid exceeds this many bins, coarsen discretization
            automatically. Default: 10,000,000.

    Example::

        # Use coarser discretization for faster computation
        acc.set_discretization(discretization=1e-3)

        # Use maximum precision
        acc.set_discretization(discretization=1e-5, max_grid_size=100_000_000)
    """
    global _default_config
    _default_config = DiscretizationConfig(
        discretization=discretization,
        log_x_mass_truncation_bound=log_x_mass_truncation_bound,
        pessimistic_estimate=pessimistic_estimate,
        max_grid_size=max_grid_size,
    )


def get_discretization(
    *,
    discretization: float | None = None,
    log_x_mass_truncation_bound: float | None = None,
    pessimistic_estimate: bool | None = None,
    max_grid_size: int | None = None,
) -> DiscretizationConfig:
    """Get discretization config with hierarchical resolution.

    Resolution priority: query param > global default > library default.
    Always returns a concrete config (never None).

    When called with no arguments, returns the current global default
    (or library default if none set). When called with arguments,
    returns a config with those overrides applied.

    Args:
        discretization: Grid spacing (query-time override).
        log_x_mass_truncation_bound: Log tail mass cutoff in x-space (query-time override).
        pessimistic_estimate: Whether to use pessimistic rounding (query-time override).
        max_grid_size: Maximum grid size before coarsening (query-time override).

    Returns:
        Resolved DiscretizationConfig (always concrete, never None).

    Example::

        # Get current global default (or library default)
        cfg = acc.get_discretization()

        # Get config with query-time overrides
        cfg = acc.get_discretization(discretization=1e-3, log_x_mass_truncation_bound=-40.0)
    """
    # Start with global default or library default
    base = _default_config if _default_config is not None else DiscretizationConfig()

    # If no overrides, return base as-is
    if (
        discretization is None
        and log_x_mass_truncation_bound is None
        and pessimistic_estimate is None
        and max_grid_size is None
    ):
        return base

    # Apply query-time overrides
    return DiscretizationConfig(
        discretization=discretization if discretization is not None else base.discretization,
        log_x_mass_truncation_bound=(
            log_x_mass_truncation_bound
            if log_x_mass_truncation_bound is not None
            else base.log_x_mass_truncation_bound
        ),
        pessimistic_estimate=(
            pessimistic_estimate if pessimistic_estimate is not None else base.pessimistic_estimate
        ),
        max_grid_size=max_grid_size if max_grid_size is not None else base.max_grid_size,
    )


