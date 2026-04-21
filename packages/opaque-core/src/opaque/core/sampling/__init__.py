"""Sampling strategies for DP-SGD training.

This module provides PyTorch-compatible samplers for differential privacy,
including Poisson sampling for privacy amplification and cyclic sampling
for matrix factorization mechanisms (BandMF).

For distributed training, shard the dataset externally using
``sampling.distributed.local_shard()`` and pass a per-rank key
via ``fold_in(key, rank)``.
"""

from opaque.core.sampling import distributed
from opaque.core.sampling._utils import PartitionType
from opaque.core.sampling.balls_in_bins import BallsInBinsSampler
from opaque.core.sampling.collate import poisson_collate
from opaque.core.sampling.b_min_sep import BMinSepSampler
from opaque.core.sampling.cyclic_poisson import CyclicPoissonSampler
from opaque.core.sampling.poisson import PoissonSampler
from opaque.core.sampling.sequential import SequentialBatchSampler
from opaque.core.sampling.truncated_poisson import TruncatedPoissonSampler

__all__ = [
    "BallsInBinsSampler",
    "PoissonSampler",
    "SequentialBatchSampler",
    "TruncatedPoissonSampler",
    "BMinSepSampler",
    "CyclicPoissonSampler",
    "PartitionType",
    "distributed",
    "poisson_collate",
]
