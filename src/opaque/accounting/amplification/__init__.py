"""Privacy amplification via subsampling.

Amplification combinators wrap a base mechanism (e.g. Gaussian) with a
sampling strategy, producing tighter privacy guarantees because each record
participates with probability < 1.

- :func:`poisson` — standard Poisson subsampling (each record sampled independently)
- :func:`truncated_poisson` — production DP-SGD with capped batch size
- :func:`accumulate` — gradient accumulation (microbatching)
"""

from opaque.accounting.amplification.accumulated import (
    Accumulated,
    accumulate,
)
from opaque.accounting.amplification.poisson import (
    Poisson,
    poisson,
)
from opaque.accounting.amplification.truncated_poisson import (
    TruncatedPoisson,
    truncated_poisson,
)

__all__ = [
    # Dataclass types
    "Poisson",
    "TruncatedPoisson",
    "Accumulated",
    # Constructor functions
    "poisson",
    "truncated_poisson",
    "accumulate",
]
