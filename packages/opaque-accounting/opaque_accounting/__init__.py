"""Differential privacy accounting using Privacy Loss Distributions (PLD).

This module provides a compositional API for tracking privacy guarantees:

- **Mechanisms**: gaussian(), band_mf(), blt_mf(), dense_mf(), etc.
- **Amplification**: poisson(), cyclic_poisson(), truncated_poisson(), etc.
- **Composition**: Combine processes using ``*`` (repeat) or ``|`` (compose)
- **Metrics**: Query privacy with epsilon_at(), delta_at(), advantage(), etc.

The underlying implementation uses Google's PLD accounting via the
``opaque-accounting`` Rust crate (PyO3 bindings).

Example::

    import opaque.accounting as acc

    # Standard DP-SGD step
    step = acc.poisson(acc.gaussian(1.1), sample_rate=0.01)
    training = step * 1000
    epsilon = training.epsilon_at(1e-5)

    # BandMF with cyclic Poisson amplification
    proc = acc.cyclic_poisson(acc.band_mf(1.0, 1000, 10), sample_rate=0.01)
    eps = proc.epsilon_at(1e-5)

For calibration (finding noise for target privacy budget), use the
:mod:`opaque.accounting.calibration` submodule.
"""

# Import native module
try:
    import opaque_accounting as _native  # noqa: F401
except ImportError as e:
    raise ImportError(
        "opaque-accounting native module not found. "
        "Build with: uv run maturin develop --release "
        "-m crates/dp-accounting/Cargo.toml"
    ) from e

# Base

# Import submodules for re-export
from . import (
    accountant,
    amplification,
    calibration,
    composition,
    mechanisms,
    transformations,
)

# Accountant
from opaque_accounting.accountant import Accountant

# Amplification
from opaque_accounting.amplification import (
    cyclic_poisson,
    parallel_poisson,
    poisson,
    truncated_poisson,
)

# Calibration
from opaque_accounting.calibration import (
    advantage_budget,
    beta_budget,
    calibrate,
    delta_budget,
    epsilon_budget,
    risk_budget,
)

# Composition
from opaque_accounting.composition import (
    cached,
    compose,
    repeat,
)
from opaque_accounting.discretization import (
    DiscretizationConfig,
    get_discretization,
    set_discretization,
)

# Mechanisms
from opaque_accounting.mechanisms import (
    band_mf,
    blt_mf,
    dense_mf,
    eps_delta,
    gaussian,
    identity,
    rectified_gaussian,
    truncated_gaussian,
)

# Transformations
from opaque_accounting.transformations import adaclip

__all__ = [
    # Submodules
    "accountant",
    "amplification",
    "calibration",
    "composition",
    "mechanisms",
    "transformations",
    # Accountant
    "Accountant",
    # Discretization
    "DiscretizationConfig",
    "set_discretization",
    "get_discretization",
    # Mechanisms (factories only; classes via subpackage import)
    "gaussian",
    "rectified_gaussian",
    "truncated_gaussian",
    "eps_delta",
    "identity",
    "band_mf",
    "blt_mf",
    "dense_mf",
    # Amplification
    "poisson",
    "truncated_poisson",
    "parallel_poisson",
    "cyclic_poisson",
    # Transformations
    "adaclip",
    # Composition
    "repeat",
    "compose",
    "cached",
    # Calibration
    "epsilon_budget",
    "delta_budget",
    "advantage_budget",
    "beta_budget",
    "risk_budget",
    "calibrate",
]
