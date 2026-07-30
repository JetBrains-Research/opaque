"""Core PLD accounting — native extension, algebra primitives, and process types.

The opaque-accounting wheel ships its impl here; users access the same
surface through the ``opaque.accounting`` façade.

Native PyO3 extension lands at
``opaque.api.accounting.core.opaque_accounting`` (matches the maturin
``module-name`` setting); aliased to ``_native`` so submodules can use
a short private name.
"""

# Native PyO3 extension — compiled artifact lives at
# ``opaque/api/accounting/core/opaque_accounting.abi3.so`` (named after
# the Rust crate). Aliased to ``_native`` for use across submodules.
try:
    from . import opaque_accounting as _native  # noqa: F401
except ImportError as e:
    raise ImportError(
        "opaque.api.accounting.core native extension not found. "
        "Build with: uv run maturin develop --release "
        "-m packages/opaque-accounting/Cargo.toml"
    ) from e

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("opaque-accounting")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Side-effect import: registers Accountant + DpProcess subclasses with
# the unified ``opaque.api.base.serialization`` registry.
import opaque.api.accounting.core._serialization  # noqa: F401
from opaque.api.accounting.core._accountant import Accountant
from opaque.api.accounting.core.calibration import (
    advantage_budget,
    beta_budget,
    calibrate,
    delta_budget,
    epsilon_budget,
    risk_budget,
)
from opaque.api.accounting.core.composition import cached, compose, repeat
from opaque.api.accounting.core.discretization import (
    get_discretization,
    set_discretization,
)
from opaque.api.accounting.core.mechanisms import eps_delta, identity, nonprivate

from . import (
    amplification,
    calibration,
    composition,
    discretization,
    mechanisms,
)

__all__ = [
    # Accountant
    "Accountant",
    "__version__",
    "advantage_budget",
    # Submodules
    "amplification",
    "beta_budget",
    "cached",
    "calibrate",
    "calibration",
    "compose",
    "composition",
    "delta_budget",
    "discretization",
    # Generic mechanisms
    "eps_delta",
    # Calibration / budgets
    "epsilon_budget",
    "get_discretization",
    "identity",
    "mechanisms",
    "nonprivate",
    # Composition
    "repeat",
    "risk_budget",
    # Discretization
    "set_discretization",
]
