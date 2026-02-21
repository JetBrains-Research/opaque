"""Base mechanism types and constructors for DP processes.

Each mechanism is a frozen dataclass (:class:`DpProcess` subclass) storing
its parameters.  The PLD is computed on demand via ``pld()`` — each call
recomputes from scratch.  Use :func:`~opaque.accounting.composition.cached`
to memoize.

Constructor functions (e.g. ``gaussian()``) validate inputs,
resolve discretization config, and return the appropriate type.

For subsampling amplification (Poisson, truncated Poisson, parallel Poisson),
see :mod:`opaque.accounting.amplification`.
"""

from opaque.accounting.mechanisms.eps_delta import (
    EpsDelta,
    eps_delta,
)
from opaque.accounting.mechanisms.gaussian import (
    Gaussian,
    gaussian,
)
from opaque.accounting.mechanisms.identity import (
    Identity,
    identity,
)
from opaque.accounting.mechanisms.mf_gaussian import (
    MfGaussian,
    mf_gaussian,
)
from opaque.accounting.mechanisms.rectified_gaussian import (
    RectifiedGaussian,
    rectified_gaussian,
)
from opaque.accounting.mechanisms.truncated_gaussian import (
    TruncatedGaussian,
    truncated_gaussian,
)

__all__ = [
    # Dataclass types
    "Gaussian",
    "EpsDelta",
    "Identity",
    "MfGaussian",
    "RectifiedGaussian",
    "TruncatedGaussian",
    # Constructor functions
    "gaussian",
    "eps_delta",
    "identity",
    "mf_gaussian",
    "rectified_gaussian",
    "truncated_gaussian",
]
