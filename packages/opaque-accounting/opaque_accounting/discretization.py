"""Private module — builds native DiscretizationConfig from keyword arguments."""

from __future__ import annotations

from . import opaque_accounting as _native

# Default values matching Google's dp_accounting library.
_DEFAULT_DISCRETIZATION: float = 1e-4
_DEFAULT_LOG_MASS_TRUNCATION_BOUND: float = -50.0
_DEFAULT_PESSIMISTIC_ESTIMATE: bool = True
_DEFAULT_MAX_GRID_SIZE: int = 10_000_000


def _make_native_config(
    discretization: float = _DEFAULT_DISCRETIZATION,
    log_mass_truncation_bound: float = _DEFAULT_LOG_MASS_TRUNCATION_BOUND,
    pessimistic_estimate: bool = _DEFAULT_PESSIMISTIC_ESTIMATE,
    max_grid_size: int = _DEFAULT_MAX_GRID_SIZE,
) -> _native.DiscretizationConfig:
    """Build a native DiscretizationConfig from keyword arguments.

    This is the single place that converts Python kwargs into the Rust
    FFI config object.
    """
    return _native.DiscretizationConfig(
        discretization=discretization,
        log_mass_truncation_bound=log_mass_truncation_bound,
        pessimistic_estimate=pessimistic_estimate,
        max_grid_size=max_grid_size,
    )
