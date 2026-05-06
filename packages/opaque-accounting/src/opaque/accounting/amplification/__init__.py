"""Privacy amplification via subsampling.

Amplification combinators wrap a base mechanism (e.g. Gaussian, BandMf) with
a sampling strategy, producing tighter privacy guarantees because each record
participates with probability < 1.

- :func:`poisson` — standard Poisson subsampling (each record sampled independently)
- :func:`truncated_poisson` — production DP-SGD with capped batch size
- :func:`parallel_poisson` — Poisson subsampling under parallel worker execution
- :func:`cyclic_poisson` — cyclic Poisson subsampling for BandMF amplification
- :func:`b_min_sep` — warm-start b-min-sep subsampling for BandMF (Monte Carlo PLD)
- :func:`balls_in_bins` — Balls-in-Bins partitioning (exact once-per-epoch participation)

The amplification dataclasses (``Poisson``, ``BallsInBins``, …) live in
:mod:`opaque.accounting.amplification.types`.
"""

from opaque.accounting.amplification._balls_in_bins import balls_in_bins
from opaque.accounting.amplification._b_min_sep import b_min_sep
from opaque.accounting.amplification._cyclic_poisson import cyclic_poisson
from opaque.accounting.amplification._parallel_poisson import parallel_poisson
from opaque.accounting.amplification._poisson import poisson
from opaque.accounting.amplification._truncated_poisson import truncated_poisson

__all__ = [
    "balls_in_bins",
    "poisson",
    "truncated_poisson",
    "parallel_poisson",
    "cyclic_poisson",
    "b_min_sep",
]
