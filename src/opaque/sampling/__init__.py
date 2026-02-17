"""Sampling strategies for DP-SGD training.

This module provides PyTorch-compatible samplers for differential privacy,
including Poisson sampling for privacy amplification and cyclic sampling
for matrix factorization mechanisms (BandMF).

Supports distributed training with two sampling modes:
- INDEPENDENT: Each worker samples independently (default for single device)
- SHARDED: Workers sample from disjoint shards (default for distributed, ensures "single Poisson")

Note: PoissonSampler and TruncatedPoissonSampler support distributed training
with automatic environment detection. CyclicPoissonSampling is for BandMF and
does not currently support distributed training.
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
from opaque.sampling.types import SamplingMode

__all__ = [
    "PoissonSampler",
    "TruncatedPoissonSampler",
    "SamplingMode",
    "BatchSelectionStrategy",
    "CyclicPoissonSampling",
    "PartitionType",
    "split_and_pad_global_batch",
    "pad_to_multiple_of",
]
