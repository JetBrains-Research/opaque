"""Private module for PLD discretization configuration management."""

from __future__ import annotations

from collections.abc import Mapping

import opaque_accounting as _native

DiscretizationConfig = _native.DiscretizationConfig

__all__ = [
    "DiscretizationConfig",
    "set_discretization",
    "get_discretization",
    "resolve_pld_config",
    "serialize_config",
    "deserialize_config",
]

# Module-level discretization default
_default_config: DiscretizationConfig | None = None


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


def get_discretization() -> DiscretizationConfig | None:
    """Get the current module-level default discretization config.

    Returns:
        Current DiscretizationConfig, or None if using native defaults.
    """
    return _default_config


def resolve_pld_config(
    config: None | float | DiscretizationConfig,
) -> DiscretizationConfig | None:
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
        # Use module default as base (if set), overriding only discretization.
        if _default_config is not None:
            return DiscretizationConfig(
                discretization=float(config),
                log_mass_truncation_bound=_default_config.log_mass_truncation_bound,
                pessimistic_estimate=_default_config.pessimistic_estimate,
                max_grid_size=_default_config.max_grid_size,
            )
        return DiscretizationConfig(discretization=float(config))
    else:
        return config

def serialize_config(
    config: DiscretizationConfig | None,
) -> dict[str, float | int | bool] | None:
    """Serialize a DiscretizationConfig to a plain dict, or None."""
    if config is None:
        return None
    return {
        "discretization": config.discretization,
        "log_mass_truncation_bound": config.log_mass_truncation_bound,
        "pessimistic_estimate": config.pessimistic_estimate,
        "max_grid_size": config.max_grid_size,
    }


def deserialize_config(
    data: Mapping[str, float | int | bool] | None,
) -> DiscretizationConfig | None:
    """Deserialize a DiscretizationConfig from a dict, or return None."""
    if data is None:
        return None
    return DiscretizationConfig(
        discretization=float(data["discretization"]),
        log_mass_truncation_bound=float(data["log_mass_truncation_bound"]),
        pessimistic_estimate=bool(data["pessimistic_estimate"]),
        max_grid_size=int(data["max_grid_size"]),
    )