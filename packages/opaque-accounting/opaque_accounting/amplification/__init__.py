"""Privacy amplification via subsampling.

Amplification combinators wrap a base mechanism (e.g. Gaussian, BandMf) with a
sampling strategy, producing tighter privacy guarantees because each record
participates with probability < 1.

- :func:`poisson` — standard Poisson subsampling (each record sampled independently)
- :func:`truncated_poisson` — production DP-SGD with capped batch size
- :func:`parallel_poisson` — Poisson subsampling under parallel worker execution
- :func:`cyclic_poisson` — cyclic Poisson subsampling for BandMF amplification
- :func:`balls_in_bins` — Balls-in-Bins partitioning (exact once-per-epoch participation)
- :func:`random_allocation_pld` — random allocation from a base PLD (deterministic path)
"""

from opaque_accounting.amplification.balls_in_bins import (
    BallsInBins,
    balls_in_bins,
)
from opaque_accounting.amplification.cyclic_poisson import (
    CyclicPoisson,
    cyclic_poisson,
)
from opaque_accounting.amplification.parallel_poisson import (
    ParallelPoisson,
    parallel_poisson,
)
from opaque_accounting.amplification.poisson import (
    Poisson,
    poisson,
)
from opaque_accounting.amplification.random_allocation import random_allocation_pld
from opaque_accounting.amplification.truncated_poisson import (
    TruncatedPoisson,
    truncated_poisson,
)

__all__ = [
    # Dataclass types
    "BallsInBins",
    "Poisson",
    "TruncatedPoisson",
    "ParallelPoisson",
    "CyclicPoisson",
    # Constructor functions
    "balls_in_bins",
    "random_allocation_pld",
    "poisson",
    "truncated_poisson",
    "parallel_poisson",
    "cyclic_poisson",
]
