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

# Base
from opaque.accounting.base import DpProcess
from opaque.accounting.discretization import (
    DiscretizationConfig,
    get_discretization,
    set_discretization,
)

# Mechanisms
from opaque.accounting.mechanisms import (
    eps_delta,
    gaussian,
    identity,
)

# Amplification
from opaque.accounting.amplification import (
    accumulate,
    poisson,
    truncated_poisson,
)

# Transformations
from opaque.accounting.transformations import adaclip

# Composition
from opaque.accounting.composition import (
    cached,
    compose,
    repeat,
)

# Accountant
from opaque.accounting.accountant import Accountant

# Calibration
from opaque.accounting.calibration import (
    advantage_budget,
    beta_budget,
    calibrate,
    delta_budget,
    epsilon_budget,
    risk_budget,
)

__all__ = [
    # Base
    "DpProcess",
    "DiscretizationConfig",
    "set_discretization",
    "get_discretization",
    # Mechanisms (factories only; classes via subpackage import)
    "gaussian",
    "eps_delta",
    "identity",
    # Amplification
    "poisson",
    "truncated_poisson",
    "accumulate",
    # Transformations
    "adaclip",
    # Composition
    "repeat",
    "compose",
    "cached",
    # Accountant
    "Accountant",
    # Calibration
    "epsilon_budget",
    "delta_budget",
    "advantage_budget",
    "beta_budget",
    "risk_budget",
    "calibrate",
]
