"""Public type definitions for :mod:`opaque.accounting`.

One-stop import for every dataclass / state / config type in the
accounting package, including the interactive :class:`Accountant`
container.  The functional surface (``gaussian()``, ``poisson()``,
``calibrate()``, etc.) lives in the package init.

For narrower namespaces, types are also re-exported from per-subpackage
``types`` modules:

- :mod:`opaque.accounting.mechanisms.types` — mechanism dataclasses
- :mod:`opaque.accounting.amplification.types` — subsampling dataclasses
- :mod:`opaque.accounting.composition.types` — composition-node types
- :mod:`opaque.accounting.transformations.types` — transformation types

DP-SGD-specific dataclasses (``Gaussian``, ``Poisson``, ``TruncatedPoisson``,
``ParallelPoisson``, ``AdaClip``) are also re-exported from
:mod:`opaque.dpsgd.accounting.types`; DP-FTRL-specific dataclasses
(``BandMf``, ``Blt``, ``LambdaCgd``, ``Bisr``, ``Bsr``, ``MfGaussian``,
``CyclicPoisson``, ``BMinSep``) from :mod:`opaque.dpftrl.accounting.types`.
"""

from __future__ import annotations

from opaque.accounting._accountant import Accountant
from opaque.accounting.amplification.types import (
    BallsInBins,
    BMinSep,
    CyclicPoisson,
    ParallelPoisson,
    Poisson,
    TruncatedPoisson,
)
from opaque.accounting._base import DpProcess
from opaque.accounting._budgets import (
    AdvantageBudget,
    BetaBudget,
    Budget,
    DeltaBudget,
    EpsilonBudget,
    RiskBudget,
)
from opaque.accounting.calibration import CalibrateResult
from opaque.accounting.composition.types import CachedProcess, Composed, Repeated
from opaque.accounting.discretization import DiscretizationConfig
from opaque.accounting.mechanisms.types import (
    BandMf,
    Bisr,
    Blt,
    Bsr,
    EpsDelta,
    Gaussian,
    Identity,
    LambdaCgd,
    MfGaussian,
    NonPrivate,
)
from opaque.accounting.transformations.types import AdaClip, SecondMoment

__all__ = [
    # Interactive container
    "Accountant",
    # Algebra base
    "DpProcess",
    # Budgets
    "Budget",
    "EpsilonBudget",
    "DeltaBudget",
    "AdvantageBudget",
    "BetaBudget",
    "RiskBudget",
    # Calibration / discretization
    "CalibrateResult",
    "DiscretizationConfig",
    # Mechanisms
    "Gaussian",
    "EpsDelta",
    "Identity",
    "NonPrivate",
    "MfGaussian",
    "BandMf",
    "Blt",
    "LambdaCgd",
    "Bisr",
    "Bsr",
    # Amplification
    "BallsInBins",
    "Poisson",
    "TruncatedPoisson",
    "ParallelPoisson",
    "BMinSep",
    "CyclicPoisson",
    # Composition
    "Composed",
    "Repeated",
    "CachedProcess",
    # Transformations
    "AdaClip",
    "SecondMoment",
]
