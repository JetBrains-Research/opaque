"""Sampling strategies for DP-SGD training.

This module provides PyTorch-compatible samplers for differential privacy,
including Poisson sampling for privacy amplification and cyclic sampling
for matrix factorization mechanisms (BandMF).
"""

from opaque.sampling.cyclic_poisson import (
    BatchSelectionStrategy,
    CyclicPoissonSampling,
    PartitionType,
    pad_to_multiple_of,
    split_and_pad_global_batch,
)
from opaque.sampling.poisson import PoissonSampler
from opaque.sampling.truncated_poisson import TruncatedPoissonSampler

__all__ = [
    "PoissonSampler",
    "TruncatedPoissonSampler",
    "BatchSelectionStrategy",
    "CyclicPoissonSampling",
    "PartitionType",
    "split_and_pad_global_batch",
    "pad_to_multiple_of",
]
