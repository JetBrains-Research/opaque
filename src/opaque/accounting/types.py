"""Backward-compatibility shim: re-exports from mechanisms.

.. deprecated::
    Import from ``opaque.accounting.mechanisms`` instead.
"""

from opaque.accounting.mechanisms import (  # noqa: F401
    Accumulated,
    EpsDelta,
    Gaussian,
    Poisson,
    TruncatedPoisson,
)

__all__ = ["Gaussian", "Poisson", "TruncatedPoisson", "Accumulated", "EpsDelta"]
