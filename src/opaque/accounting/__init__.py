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
    step = acc.poisson(noise_multiplier=1.1, sample_rate=0.01)

    # Compose 1000 steps
    training = step * 1000

    # Query privacy at delta=1e-5
    epsilon = training.epsilon_at(1e-5)
    print(f"Privacy: (ε={epsilon:.2f}, δ=1e-5)")

For calibration (finding noise for target privacy budget), use the
:mod:`opaque.accounting.calibration` submodule.
"""

# Import native module and types
try:
    import opaque_accounting as _native
except ImportError as e:
    raise ImportError(
        "opaque-accounting native module not found. "
        "Install with: maturin develop -m crates/dp-accounting/Cargo.toml"
    ) from e

# Re-export types
DpProcess = _native.DpProcess
"""A differential privacy process that can be queried for privacy guarantees.

This is the central class in ``opaque.accounting``. Every mechanism constructor
(``gaussian``, ``poisson``, etc.) returns a ``DpProcess``, and composition
operators produce new ``DpProcess`` instances.

All privacy metrics are derived from the same Privacy Loss Distribution (PLD):

- **epsilon_at(delta)**: Get epsilon for given delta (ε,δ-DP)
- **delta_at(epsilon)**: Get delta for given epsilon (ε,δ-DP)
- **advantage()**: Get f-DP total-variation advantage
- **beta_at(alpha)**: Get Type-II error at given Type-I error (hypothesis testing)
- **risk_at(prior)**: Get Bayes risk at given prior

Composition operators:

- **step * 1000**: Repeat a process 1000 times (homogeneous composition)
- **a | b**: Compose two different processes (heterogeneous composition)

Debugging:

- **print(proc)**: One-line summary with epsilon
- **describe()**: Constructor parameters as dict
- **pld_info()**: PLD grid diagnostics with timing
- **summary()**: Multi-line formatted privacy report

Example::

    step = acc.poisson(1.1, 0.01)
    training = step * 1000
    eps = training.epsilon_at(1e-5)
    print(training.summary())  # detailed report
"""

DiscretizationConfig = _native.DiscretizationConfig
"""Configuration controlling PLD discretization precision.

The PLD is represented as a discrete probability mass function (PMF) on a
regular grid. These parameters control grid resolution, tail truncation,
and rounding direction.

Defaults are chosen for high accuracy (discretization=1e-4 gives ~1e-8 error
per composition step). Coarser grids are faster but less precise.

Args:
    discretization: Grid spacing for PLD PMF. Default: 1e-4.
        Smaller = more precise, larger grid. Error scales as O(disc^2).
    log_mass_truncation_bound: Tails with probability below exp(bound) are
        truncated. Default: -50 (matching Google's dp_accounting).
    pessimistic_estimate: If True (default), round probabilities upward to
        produce an **upper bound** on privacy loss. If False, round downward
        (optimistic estimate - not safe for guarantees).
    max_grid_size: If grid exceeds this many bins, coarsen discretization
        automatically. Default: 10,000,000.

Example::

    # Faster but less precise
    cfg = acc.DiscretizationConfig(discretization=1e-3)

    # Maximum precision
    cfg = acc.DiscretizationConfig(
        discretization=1e-5,
        log_mass_truncation_bound=-50.0,
    )

    # Use with any mechanism
    proc = acc.gaussian(1.1, discretization=cfg)
"""

# Import discretization utilities
from opaque.accounting._discretization import (
    get_discretization,
    set_discretization,
)

# Import mechanism constructors
from opaque.accounting.mechanisms import (
    accumulate,
    adaclip,
    eps_delta,
    gaussian,
    identity,
    poisson,
    truncated_poisson,
)

# Import composition operators
from opaque.accounting.composition import (
    compose,
    repeat,
)

# Import Accountant
from opaque.accounting.accountant import Accountant

# Import calibration utilities
from opaque.accounting.calibration import (
    epsilon,
    delta,
    advantage,
    beta,
    risk,
    calibrate,
)

__all__ = [
    # Types
    "DpProcess",
    "DiscretizationConfig",
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
    # Accounting
    "Accountant",
    # Calibration targets
    "epsilon",
    "delta",
    "advantage",
    "beta",
    "risk",
    # Calibration functions
    "calibrate",
]
