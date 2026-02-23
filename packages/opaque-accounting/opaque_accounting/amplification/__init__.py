"""Privacy amplification via subsampling.

Amplification combinators wrap a base mechanism (e.g. Gaussian, BandMf) with a
sampling strategy, producing tighter privacy guarantees because each record
participates with probability < 1.

- :func:`poisson` — standard Poisson subsampling (each record sampled independently)
- :func:`truncated_poisson` — production DP-SGD with capped batch size
- :func:`parallel_poisson` — Poisson subsampling under parallel worker execution
- :func:`cyclic_poisson` — cyclic Poisson subsampling for BandMF amplification
"""

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
from opaque_accounting.amplification.truncated_poisson import (
    TruncatedPoisson,
    truncated_poisson,
)

__all__ = [
    # Dataclass types
    "Poisson",
    "TruncatedPoisson",
    "ParallelPoisson",
    "CyclicPoisson",
    # Constructor functions
    "poisson",
    "truncated_poisson",
    "parallel_poisson",
    "cyclic_poisson",
]
