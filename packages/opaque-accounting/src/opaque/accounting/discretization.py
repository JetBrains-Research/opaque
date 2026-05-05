"""Private module for PLD discretization configuration management."""

from __future__ import annotations

from dataclasses import dataclass

from . import _native


@dataclass(frozen=True, slots=True)
class DiscretizationConfig:
    """Discretization configuration for PLD computation.

    Args:
        discretization: Grid spacing for the PLD PMF. Smaller = tighter, slower.
        log_x_mass_truncation_bound: Log tail mass cutoff in x-space. Tails below exp(bound) are truncated.
        pessimistic_estimate: Round upward for safe upper bounds (True) or downward (False).
        max_grid_size: Maximum grid bins before automatic coarsening.
        num_mc_samples: Number of Monte Carlo samples for MC-based accounting.
        seed: RNG seed for Monte Carlo reproducibility.
    """

    discretization: float = 1e-4
    log_x_mass_truncation_bound: float = -50.0
    pessimistic_estimate: bool = True
    max_grid_size: int = 10_000_000
    tail_mass_truncation: float = 1e-15
    num_mc_samples: int = 100_000
    seed: int = 42

    def to_native(self) -> _native.DiscretizationConfig:
        """Convert to Rust DiscretizationConfig for FFI calls."""
        return _native.DiscretizationConfig(
            discretization=self.discretization,
            log_mass_truncation_bound=self.log_x_mass_truncation_bound,
            pessimistic_estimate=self.pessimistic_estimate,
            max_grid_size=self.max_grid_size,
            tail_mass_truncation=self.tail_mass_truncation,
            num_mc_samples=self.num_mc_samples,
            seed=self.seed,
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
    tail_mass_truncation: float = 1e-15,
    num_mc_samples: int = 100_000,
    seed: int = 42,
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
        tail_mass_truncation: Chernoff tail budget during composition (Rust default 1e-15).
        num_mc_samples: Number of Monte Carlo samples for MC-based accounting. Default: 100,000.
        seed: RNG seed for Monte Carlo reproducibility. Default: 42.

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
        tail_mass_truncation=tail_mass_truncation,
        num_mc_samples=num_mc_samples,
        seed=seed,
    )


def get_discretization(
    *,
    discretization: float | None = None,
    log_x_mass_truncation_bound: float | None = None,
    pessimistic_estimate: bool | None = None,
    max_grid_size: int | None = None,
    tail_mass_truncation: float | None = None,
    num_mc_samples: int | None = None,
    seed: int | None = None,
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
        tail_mass_truncation: Composition tail budget (query-time override).
        num_mc_samples: Number of Monte Carlo samples (query-time override).
        seed: RNG seed for Monte Carlo (query-time override).

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
        and tail_mass_truncation is None
        and num_mc_samples is None
        and seed is None
    ):
        return base

    # Apply query-time overrides
    return DiscretizationConfig(
        discretization=discretization
        if discretization is not None
        else base.discretization,
        log_x_mass_truncation_bound=(
            log_x_mass_truncation_bound
            if log_x_mass_truncation_bound is not None
            else base.log_x_mass_truncation_bound
        ),
        pessimistic_estimate=(
            pessimistic_estimate
            if pessimistic_estimate is not None
            else base.pessimistic_estimate
        ),
        max_grid_size=max_grid_size
        if max_grid_size is not None
        else base.max_grid_size,
        tail_mass_truncation=tail_mass_truncation
        if tail_mass_truncation is not None
        else base.tail_mass_truncation,
        num_mc_samples=num_mc_samples
        if num_mc_samples is not None
        else base.num_mc_samples,
        seed=seed if seed is not None else base.seed,
    )
