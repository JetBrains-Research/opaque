"""Sampling strategies for DP-SGD training.

This module provides PyTorch-compatible samplers for differential privacy,
including Poisson sampling for privacy amplification and cyclic sampling
for matrix factorization mechanisms (BandMF).

For distributed training, shard the dataset externally using
``sampling.distributed.local_shard()`` and pass a per-rank key
via ``fold_in(key, rank)``.
"""

from opaque.sampling import distributed
from opaque.sampling._utils import PartitionType
from opaque.sampling.collate import poisson_collate
from opaque.sampling.cyclic_poisson import CyclicPoissonSampler
from opaque.sampling.poisson import PoissonSampler
from opaque.sampling.truncated_poisson import TruncatedPoissonSampler

__all__ = [
    "PoissonSampler",
    "TruncatedPoissonSampler",
    "CyclicPoissonSampler",
    "PartitionType",
    "distributed",
    "poisson_collate",
]
