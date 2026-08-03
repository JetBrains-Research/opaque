"""Type definitions for :mod:`opaque.dpsgd.accounting`.

Re-exports DP-SGD-specific dataclasses for type annotations.  The
constructor functions live in the package init.
"""

from __future__ import annotations

from opaque.api.accounting.dpsgd.amplification.types import (
    KOutOfT,
    ParallelPoisson,
    Poisson,
    RandomAllocation,
)
from opaque.api.accounting.dpsgd.mechanisms.types import AdaClip, Gaussian

__all__ = [
    "AdaClip",
    "Gaussian",
    "KOutOfT",
    "ParallelPoisson",
    "Poisson",
    "RandomAllocation",
]
