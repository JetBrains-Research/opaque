"""Differential privacy accounting using Privacy Loss Distributions (PLD).

Cross-cutting accounting surface — composition, calibration, generic
mechanisms (``identity``, ``nonprivate``, ``eps_delta``), and the two
amplification / transformation primitives that span DP-SGD and DP-FTRL
(``balls_in_bins``, ``second_moment``).

Algorithm-specific factories live in dedicated namespaces:

- :mod:`opaque.dpsgd.accounting` — ``gaussian``, ``adaclip``, ``poisson``,
  ``truncated_poisson``, ``parallel_poisson``.
- :mod:`opaque.dpftrl.accounting` — ``band_mf``, ``blt``, ``bisr``,
  ``bsr``, ``lambda_cgd``, ``cyclic_poisson``, ``b_min_sep``.

Both algorithm-specific namespaces re-export from the same shared
implementation underneath; the split is purely organisational and
algorithm-specific factories remain accessible from this root module
for legacy code (the names below are imported but not in ``__all__``).

Implementation uses Google's PLD accounting via the ``opaque-accounting``
Rust crate (PyO3 bindings).

Example::

    import opaque.dpsgd.accounting as dpsgd_acc

    step = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), sample_rate=0.01)
    training = step * 1000
    epsilon = training.epsilon_at(1e-5)

For calibration (finding noise for target privacy budget), use the
:mod:`opaque.accounting.calibration` submodule. All public dataclasses
— including the interactive :class:`Accountant` — live in
:mod:`opaque.accounting.types`, with narrower per-subpackage ``types``
modules also available.
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

# Submodules re-exported as documented power-user surface.
from . import (
    amplification,
    calibration,
    composition,
    discretization,
    mechanisms,
    transformations,
)

# ──────────────────────────────────────────────────────────────────────
# Headline (in __all__): cross-cutting + composition + calibration only.
# ──────────────────────────────────────────────────────────────────────
from opaque.accounting.amplification import balls_in_bins
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
from opaque.accounting.transformations import second_moment

# ──────────────────────────────────────────────────────────────────────
# Legacy access (NOT in __all__): algorithm-specific factories accessible
# from this root for backward compatibility.  New code should import
# from opaque.dpsgd.accounting / opaque.dpftrl.accounting.
# ──────────────────────────────────────────────────────────────────────
from opaque.accounting._accountant import Accountant  # noqa: F401
from opaque.accounting.amplification import (  # noqa: F401
    b_min_sep,
    cyclic_poisson,
    parallel_poisson,
    poisson,
    truncated_poisson,
)
from opaque.accounting.mechanisms import (  # noqa: F401
    band_mf,
    bisr,
    blt,
    bsr,
    gaussian,
    lambda_cgd,
)
from opaque.accounting.transformations import adaclip  # noqa: F401

__all__ = [
    "__version__",
    # Submodules
    "amplification",
    "calibration",
    "composition",
    "discretization",
    "mechanisms",
    "transformations",
    # Discretization
    "set_discretization",
    "get_discretization",
    # Generic mechanisms
    "eps_delta",
    "identity",
    "nonprivate",
    # Cross-cutting amplification / transformation
    "balls_in_bins",
    "second_moment",
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
