"""Sampling primitives (algorithm-agnostic).

Core exposes Poisson sampling, collation helpers, and distributed shard
helpers. Algorithm-specific samplers live with their mechanism:

- ``opaque.dpsgd.sampling.TruncatedPoissonSampler`` (fixed-batch DP-SGD)
- ``opaque.mf.sampling.{BMinSepSampler, CyclicPoissonSampler,
  BallsInBinsSampler, SequentialBatchSampler}`` (matrix factorization)

For distributed training, shard the dataset externally using
``sampling.distributed.local_shard()`` and pass a per-rank key
via ``fold_in(key, rank)``.
"""

from opaque.core.sampling import distributed
from opaque.core.sampling._utils import PartitionType
from opaque.core.sampling.collate import poisson_collate
from opaque.core.sampling.poisson import PoissonSampler

__all__ = [
    "PoissonSampler",
    "PartitionType",
    "distributed",
    "poisson_collate",
]
