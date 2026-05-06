"""Public type definitions for :mod:`opaque.accounting.amplification`.

Re-exports the subsampling-amplification dataclasses for type annotations.
The constructor functions (``poisson()``, ``balls_in_bins()``, …) live in
the package init.
"""

from __future__ import annotations

from opaque.accounting.amplification._balls_in_bins import BallsInBins
from opaque.accounting.amplification._b_min_sep import BMinSep
from opaque.accounting.amplification._cyclic_poisson import CyclicPoisson
from opaque.accounting.amplification._parallel_poisson import ParallelPoisson
from opaque.accounting.amplification._poisson import Poisson
from opaque.accounting.amplification._truncated_poisson import TruncatedPoisson

__all__ = [
    "BallsInBins",
    "Poisson",
    "TruncatedPoisson",
    "ParallelPoisson",
    "BMinSep",
    "CyclicPoisson",
]
