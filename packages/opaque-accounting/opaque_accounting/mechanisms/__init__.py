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

from opaque_accounting.mechanisms.band_mf import (
    BandMf,
    band_mf,
)
from opaque_accounting.mechanisms.blt_mf import (
    BltMf,
    blt_mf,
)
from opaque_accounting.mechanisms.dense_mf import (
    DenseMf,
    dense_mf,
)
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
from opaque_accounting.mechanisms.nonprivate import (
    NonPrivate,
    nonprivate,
)
from opaque_accounting.mechanisms.truncated_gaussian import (
    TruncatedGaussian,
    truncated_gaussian,
)

__all__ = [
    # Dataclass types
    "Gaussian",
    "EpsDelta",
    "Identity",
    "NonPrivate",
    "BandMf",
    "BltMf",
    "DenseMf",
    "TruncatedGaussian",
    # Constructor functions
    "gaussian",
    "eps_delta",
    "identity",
    "nonprivate",
    "band_mf",
    "blt_mf",
    "dense_mf",
    "truncated_gaussian",
]
