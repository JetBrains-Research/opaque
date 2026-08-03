"""DP-SGD samplers façade — Poisson subsampling and random allocation.

:class:`PoissonSampler` covers plain Poisson subsampling (default). Pass
``truncated_batch_size`` to cap batch size (truncated Poisson; weaker
privacy than plain Poisson at the same rate unless noise is recalibrated —
use matching accounting).

:class:`RandomAllocationSampler` partitions the dataset into ``num_bins``
bins every epoch, redrawing the assignment each time. It amplifies more
than Poisson at the matched rate ``1/num_bins``; pair it with
``opaque.dpsgd.accounting.random_allocation``.

:class:`KOutOfTSampler` chooses exactly ``total_participations`` steps from a
declared horizon uniformly for every record.
"""

from opaque.api.dpsgd.sampling import (
    KOutOfTSampler,
    PoissonSampler,
    RandomAllocationSampler,
)

__all__ = ["KOutOfTSampler", "PoissonSampler", "RandomAllocationSampler"]
