"""Base mechanism types and constructors for DP processes.

Each mechanism is a frozen dataclass (:class:`DpProcess` subclass) storing
its parameters.  The PLD is computed on demand via ``pld()`` — each call
recomputes from scratch.  Use :func:`~opaque.accounting.composition.cached`
to memoize.

Constructor functions (e.g. ``gaussian()``) validate inputs,
resolve discretization config, and return the appropriate type.

For subsampling amplification (Poisson, truncated Poisson, accumulated),
see :mod:`opaque.accounting.amplification`.
"""

from opaque.accounting.mechanisms.bounded_gaussian import (
    BoundedGaussian,
    bounded_gaussian,
)
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

__all__ = [
    # Dataclass types
    "BoundedGaussian",
    "Gaussian",
    "EpsDelta",
    "Identity",
    # Constructor functions
    "bounded_gaussian",
    "gaussian",
    "eps_delta",
    "identity",
]
