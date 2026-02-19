"""Private module for PLD discretization configuration management."""

from __future__ import annotations

from typing import Optional, Union

import opaque_accounting as _native

DiscretizationConfig = _native.DiscretizationConfig

# Legacy alias for backward compatibility
PldConfig = DiscretizationConfig

# Module-level discretization default
_default_config: Optional[DiscretizationConfig] = None


def set_discretization(
    discretization: float = 1e-4,
    log_mass_truncation_bound: float = -50.0,
    pessimistic_estimate: bool = True,
    max_grid_size: int = 10_000_000,
) -> None:
    """Set module-level default discretization parameters.

    These defaults are used when ``discretization=None`` is passed to mechanism
    constructors. By default, uses high-precision settings matching Google's
    ``dp_accounting`` library.

    Args:
        discretization: Grid spacing for PLD PMF. Smaller = more precise, larger = faster.
            Error scales as O(disc^2). Default: 1e-4.
        log_mass_truncation_bound: Tails with probability below exp(bound) are
            truncated. Default: -50 (matching Google).
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
        log_mass_truncation_bound=log_mass_truncation_bound,
        pessimistic_estimate=pessimistic_estimate,
        max_grid_size=max_grid_size,
    )


def get_discretization() -> Optional[DiscretizationConfig]:
    """Get the current module-level default discretization config.

    Returns:
        Current DiscretizationConfig, or None if using native defaults.
    """
    return _default_config


def resolve_pld_config(
    config: Union[None, float, DiscretizationConfig],
) -> Optional[DiscretizationConfig]:
    """Resolve discretization parameter to a DiscretizationConfig object.

    Args:
        config: None (use module default), float (use as discretization value),
            or DiscretizationConfig (use as-is).

    Returns:
        Resolved DiscretizationConfig or None (use Rust defaults).
    """
    if config is None:
        return _default_config
    elif isinstance(config, (int, float)):
        return DiscretizationConfig(discretization=float(config))
    else:
        return config


# Legacy alias
resolve_discretization = resolve_pld_config
