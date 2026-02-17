"""Private module for discretization configuration management."""

from typing import Optional, Union

try:
    import opaque_accounting as _native
except ImportError as e:
    raise ImportError(
        "opaque-accounting native module not found. "
        "Install with: maturin develop -m crates/dp-accounting/Cargo.toml"
    ) from e

DiscretizationConfig = _native.DiscretizationConfig

# Module-level discretization default
_default_discretization: Optional[DiscretizationConfig] = None


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
        pessimistic_estimate: If True (default), round probabilities upward to produce
            an **upper bound** on privacy loss (safe for guarantees). If False, round
            downward (optimistic estimate, useful for debugging only).
        max_grid_size: If grid exceeds this many bins, coarsen discretization
            automatically. Default: 10,000,000.

    Example::

        # Use coarser discretization for faster computation
        acc.set_discretization(discretization=1e-3)

        # Use maximum precision
        acc.set_discretization(discretization=1e-5, max_grid_size=100_000_000)
    """
    global _default_discretization
    _default_discretization = DiscretizationConfig(
        discretization=discretization,
        log_mass_truncation_bound=log_mass_truncation_bound,
        pessimistic_estimate=pessimistic_estimate,
        max_grid_size=max_grid_size,
    )


def get_discretization() -> Optional[DiscretizationConfig]:
    """Get the current module-level default discretization config.

    Returns:
        Current discretization config, or None if using native defaults.
    """
    return _default_discretization


def resolve_discretization(
    config: Union[None, float, DiscretizationConfig]
) -> Optional[DiscretizationConfig]:
    """Resolve discretization parameter to a config object.

    Args:
        config: None (use module default), float (use as discretization value),
            or DiscretizationConfig (use as-is).

    Returns:
        Resolved DiscretizationConfig or None (use Rust defaults).
    """
    if config is None:
        return _default_discretization
    elif isinstance(config, (int, float)):
        # Convert float to DiscretizationConfig using current defaults
        base = _default_discretization or DiscretizationConfig()
        return DiscretizationConfig(
            discretization=float(config),
            log_mass_truncation_bound=base.log_mass_truncation_bound,
            pessimistic_estimate=base.pessimistic_estimate,
            max_grid_size=base.max_grid_size,
        )
    else:
        return config
