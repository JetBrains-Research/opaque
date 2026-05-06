"""Differential privacy accounting using Privacy Loss Distributions (PLD).

Compositional API for tracking privacy guarantees:

- **Mechanisms**: ``gaussian()``, ``lambda_cgd()``, ``bisr()``, ``bsr()``,
  ``band_mf()``, ``blt()``, …
- **Amplification**: ``balls_in_bins()``, ``poisson()``, ``cyclic_poisson()``, …
- **Composition**: combine processes with ``*`` (repeat) or ``|`` (compose)
- **Metrics**: query privacy with ``epsilon_at()``, ``delta_at()``,
  ``advantage()``, …

Implementation uses Google's PLD accounting via the ``opaque-accounting``
Rust crate (PyO3 bindings).

Example::

    import opaque.accounting as acc

    # Standard DP-SGD step
    step = acc.poisson(acc.gaussian(1.1), sample_rate=0.01)
    training = step * 1000
    epsilon = training.epsilon_at(1e-5)

    # DP-λCGD with Balls-in-Bins amplification
    training = acc.balls_in_bins(
        acc.lambda_cgd(1.0, sensitivity=s.sensitivity,
                       gram_matrix=s.gram_matrix),
        num_bins=1875, num_epochs=8,
    )
    eps = training.epsilon_at(1e-5)

For calibration (finding noise for target privacy budget), use the
:mod:`opaque.accounting.calibration` submodule. All public dataclasses
(``DpProcess``, ``Budget``, mechanism / amplification / transformation /
composition node types, ``CalibrateResult``, ``DiscretizationConfig``)
live in :mod:`opaque.accounting.types`, with narrower per-subpackage
``types`` modules also available.
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

from opaque.accounting._accountant import Accountant
from opaque.accounting.amplification import (
    b_min_sep,
    balls_in_bins,
    cyclic_poisson,
    parallel_poisson,
    poisson,
    truncated_poisson,
)
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
from opaque.accounting.mechanisms import (
    band_mf,
    bisr,
    blt,
    bsr,
    eps_delta,
    gaussian,
    identity,
    lambda_cgd,
    nonprivate,
)
from opaque.accounting.transformations import adaclip, second_moment

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
    # Mechanisms
    "gaussian",
    "eps_delta",
    "identity",
    "nonprivate",
    "band_mf",
    "blt",
    "lambda_cgd",
    "bisr",
    "bsr",
    # Amplification
    "balls_in_bins",
    "poisson",
    "truncated_poisson",
    "parallel_poisson",
    "b_min_sep",
    "cyclic_poisson",
    # Transformations
    "adaclip",
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
