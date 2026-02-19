"""Differential privacy accounting using Privacy Loss Distributions (PLD).

This module provides a compositional API for tracking privacy guarantees:

- **Mechanisms**: gaussian(), poisson(), truncated_poisson(), etc.
- **Composition**: Combine processes using ``*`` (repeat) or ``|`` (compose)
- **Metrics**: Query privacy with epsilon_at(), delta_at(), advantage(), etc.

The underlying implementation uses Google's PLD accounting via the
``opaque-accounting`` Rust crate (PyO3 bindings).

Example::

    import opaque.accounting as acc

    # Create a DP-SGD step
    step = acc.poisson(acc.gaussian(1.1), sample_rate=0.01)

    # Compose 1000 steps
    training = step * 1000

    # Query privacy at delta=1e-5
    epsilon = training.epsilon_at(1e-5)
    print(f"Privacy: (ε={epsilon:.2f}, δ=1e-5)")

For calibration (finding noise for target privacy budget), use the
:mod:`opaque.accounting.calibration` submodule.
"""

# Import native module
try:
    import opaque_accounting as _native
except ImportError as e:
    raise ImportError("opaque-accounting native module not found. ") from e

# Base class
from opaque.accounting.accountant import Accountant
from opaque.accounting.base import DpProcess
from opaque.accounting.calibration import (
    advantage_budget,
    beta_budget,
    calibrate,
    delta_budget,
    epsilon_budget,
    risk_budget,
)
from opaque.accounting.composition import (
    cached,
    compose,
    repeat,
)

# Config types
from opaque.accounting.discretization import (
    DiscretizationConfig,
    get_discretization,
    set_discretization,
)
from opaque.accounting.mechanisms import (
    accumulate,
    adaclip,
    eps_delta,
    gaussian,
    identity,
    poisson,
    truncated_poisson,
)

# Legacy alias: code that imports PldConfig still works
PldConfig = DiscretizationConfig
"""Configuration controlling PLD discretization precision.

The PLD is represented as a discrete probability mass function (PMF) on a
regular grid. The ``DiscretizationConfig`` (aliased as ``PldConfig``) controls
grid resolution and tail truncation.

Args:
    discretization: Grid spacing for PLD PMF. Default: 1e-4.
        Smaller = more precise, larger grid. Error scales as O(disc^2).
    log_mass_truncation_bound: Tails with probability below exp(bound) are
        truncated. Default: -50 (matching Google's dp_accounting).

Example::

    cfg = acc.DiscretizationConfig(discretization=1e-3)
    proc = acc.gaussian(1.1, discretization=cfg)
"""

__all__ = [
    # Types
    "DpProcess",
    "DiscretizationConfig",
    "PldConfig",
    # Module defaults
    "set_discretization",
    "get_discretization",
    # Mechanisms
    "gaussian",
    "poisson",
    "truncated_poisson",
    "accumulate",
    "adaclip",
    "eps_delta",
    "identity",
    # Composition
    "repeat",
    "compose",
    "cached",
    # Accounting
    "Accountant",
    # Calibration targets
    "epsilon_budget",
    "delta_budget",
    "advantage_budget",
    "beta_budget",
    "risk_budget",
    # Calibration functions
    "calibrate",
]
