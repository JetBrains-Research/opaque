"""Differential privacy accounting using Privacy Loss Distributions (PLD).

Cross-cutting accounting surface — composition, calibration, generic
mechanisms (``identity``, ``nonprivate``, ``eps_delta``), and shared
transformations.

Algorithm-specific factories live in their respective packages
(``opaque-dpsgd`` / ``opaque-dpftrl``):

- :mod:`opaque.dpsgd.accounting` — ``gaussian``, ``adaclip``, ``poisson``,
  ``truncated_poisson``, ``parallel_poisson``.
- :mod:`opaque.dpftrl.accounting` — ``band_mf``, ``blt``, ``bisr``,
  ``bsr``, ``lambda_cgd``, ``cyclic_poisson``, ``b_min_sep``,
  ``balls_in_bins``.

Implementation uses Google's PLD accounting via the ``opaque-accounting``
Rust crate (PyO3 bindings).

Example (requires ``opaque-dpsgd`` in the environment)::

    import opaque.accounting as acc
    import opaque.dpsgd.accounting as dpsgd_acc

    step = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), sample_rate=0.01)
    training = step * 1000
    epsilon = training.epsilon_at(1e-5)
"""

# Native PyO3 extension. Compiled artifact lives at
# ``opaque/accounting/opaque_accounting.abi3.so`` (named after the Rust
# crate); aliased to ``_native`` so submodules can use a short private
# name.
try:
    from . import opaque_accounting as _native  # noqa: F401
except ImportError as e:
    raise ImportError(
        "opaque.accounting native extension not found. "
        "Build with: uv run maturin develop --release "
        "-m packages/opaque-accounting/Cargo.toml"
    ) from e

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("opaque-accounting")
except PackageNotFoundError:
    __version__ = "0.0.0"

from . import (
    amplification,
    calibration,
    composition,
    discretization,
    mechanisms,
    transformations,
)

from opaque.accounting._accountant import Accountant

import opaque.accounting._serialization  # noqa: F401  (opaque.serialization hook)
from opaque.accounting.calibration import (
    advantage_budget,
    beta_budget,
    calibrate,
    delta_budget,
    epsilon_budget,
    risk_budget,
)
from opaque.accounting.composition import cached, compose, repeat
from opaque.accounting.discretization import get_discretization, set_discretization
from opaque.accounting.mechanisms import eps_delta, identity, nonprivate

__all__ = [
    "__version__",
    # Submodules
    "amplification",
    "calibration",
    "composition",
    "discretization",
    "mechanisms",
    "transformations",
    # Accountant
    "Accountant",
    # Discretization
    "set_discretization",
    "get_discretization",
    # Generic mechanisms
    "eps_delta",
    "identity",
    "nonprivate",
    # Composition
    "repeat",
    "compose",
    "cached",
    # Calibration / budgets
    "epsilon_budget",
    "delta_budget",
    "advantage_budget",
    "beta_budget",
    "risk_budget",
    "calibrate",
]
