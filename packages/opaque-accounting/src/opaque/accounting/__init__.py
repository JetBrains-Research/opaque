"""Differential privacy accounting using Privacy Loss Distributions (PLD).

This module provides a compositional API for tracking privacy guarantees:

- **Mechanisms**: gaussian(), lambda_cgd(), bisr(), bsr(), band_mf(), blt(), etc.
- **Amplification**: balls_in_bins(), poisson(), cyclic_poisson(), etc.
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

    # DP-λCGD with Balls-in-Bins amplification
    training = acc.balls_in_bins(
        acc.lambda_cgd(1.0, sensitivity=s.sensitivity,
                       gram_matrix=s.gram_matrix),
        num_bins=1875, num_epochs=8,
    )
    eps = training.epsilon_at(1e-5)

    # BandMF with cyclic Poisson amplification
    proc = acc.cyclic_poisson(acc.band_mf(1.0, sensitivity=1.0, num_groups=100), sample_rate=0.01)
    eps = proc.epsilon_at(1e-5)

For calibration (finding noise for target privacy budget), use the
:mod:`opaque.accounting.calibration` submodule.
"""

# Import native module (PyO3 extension).
# The compiled artifact lives at `opaque/accounting/opaque_accounting.abi3.so`
# (the file name matches the Rust crate for a sensible-looking .so), but inside
# Python we alias it to ``_native`` so submodules can keep a short, conventional
# private-impl name.
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
from opaque.accounting.accountant import Accountant

# Amplification
from opaque.accounting.amplification import (
    balls_in_bins,
    b_min_sep,
    cyclic_poisson,
    parallel_poisson,
    poisson,
    truncated_poisson,
)

# Calibration
from opaque.accounting.calibration import (
    advantage_budget,
    beta_budget,
    calibrate,
    delta_budget,
    epsilon_budget,
    risk_budget,
)

# Composition
from opaque.accounting.composition import (
    cached,
    compose,
    repeat,
)
from opaque.accounting.discretization import (
    DiscretizationConfig,
    get_discretization,
    set_discretization,
)

# Mechanisms
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

# Transformations
from opaque.accounting.transformations import adaclip, jme

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
    "jme",
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
