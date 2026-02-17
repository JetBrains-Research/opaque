"""Sampling strategies for DP-SGD training.

This module provides PyTorch-compatible samplers for differential privacy,
including Poisson sampling for privacy amplification and cyclic sampling
for matrix factorization mechanisms (BandMF).

Supports distributed training with two sampling modes:
- INDEPENDENT: Each worker samples independently (default for single device)
- SHARDED: Workers sample from disjoint shards (default for distributed, ensures "single Poisson")

Note: PoissonSampler, TruncatedPoissonSampler, and CyclicPoissonSampler support
distributed training with automatic environment detection.
"""

from opaque.sampling._utils import PartitionType
from opaque.sampling.cyclic_poisson import CyclicPoissonSampler
from opaque.sampling.poisson import PoissonSampler
from opaque.sampling.truncated_poisson import TruncatedPoissonSampler
from opaque.sampling.types import SamplingMode

__all__ = [
    "PoissonSampler",
    "TruncatedPoissonSampler",
    "CyclicPoissonSampler",
    "SamplingMode",
    "PartitionType",
]
