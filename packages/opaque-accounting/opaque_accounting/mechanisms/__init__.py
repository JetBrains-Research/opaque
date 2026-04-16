"""Base mechanism types and constructors for DP processes.

Each mechanism is a frozen dataclass (:class:`DpProcess` subclass) storing
its parameters.  The PLD is computed on demand via ``pld()`` — each call
recomputes from scratch.  Use :func:`~opaque.accounting.composition.cached`
to memoize.

Constructor functions (e.g. ``gaussian()``) validate inputs,
resolve discretization config, and return the appropriate type.

For subsampling amplification (Poisson, truncated Poisson, parallel Poisson,
cyclic Poisson), see :mod:`opaque.accounting.amplification`.
"""

from opaque_accounting.mechanisms.eps_delta import (
    EpsDelta,
    eps_delta,
)
from opaque_accounting.mechanisms.gaussian import (
    Gaussian,
    gaussian,
)
from opaque_accounting.mechanisms.identity import (
    Identity,
    identity,
)
from opaque_accounting.mechanisms.band_mf import (
    BandMf,
    band_mf,
)
from opaque_accounting.mechanisms.bisr import (
    Bisr,
    bisr,
)
from opaque_accounting.mechanisms.bsr import (
    Bsr,
    bsr,
)
from opaque_accounting.mechanisms.lr_aware import (
    LrAware,
    lr_aware,
)
from opaque_accounting.mechanisms.blt import (
    Blt,
    blt,
)
from opaque_accounting.mechanisms.lambda_cgd import (
    LambdaCgd,
    lambda_cgd,
)
from opaque_accounting.mechanisms.mf_gaussian import (
    MfGaussian,
)
from opaque_accounting.mechanisms.nonprivate import (
    NonPrivate,
    nonprivate,
)

__all__ = [
    # Dataclass types
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
    "LrAware",
    # Constructor functions
    "gaussian",
    "eps_delta",
    "identity",
    "nonprivate",
    "band_mf",
    "blt",
    "lambda_cgd",
    "bisr",
    "bsr",
    "lr_aware",
]
