"""Sampling strategies for DP-SGD training.

This module provides PyTorch-compatible samplers for differential privacy,
including Poisson sampling for privacy amplification and cyclic sampling
for matrix factorization mechanisms (BandMF).

Supports distributed training with auto detection or explicit overrides:
- distributed="auto": SHARDED when distributed is initialized, otherwise INDEPENDENT
- distributed=True: force SHARDED
- distributed=False: force INDEPENDENT

Note: PoissonSampler, TruncatedPoissonSampler, and CyclicPoissonSampler support
distributed training with automatic environment detection.
"""

from opaque.sampling import distributed
from opaque.sampling._utils import PartitionType
from opaque.sampling.cyclic_poisson import CyclicPoissonSampler
from opaque.sampling.poisson import PoissonSampler
from opaque.sampling.truncated_poisson import TruncatedPoissonSampler

__all__ = [
    "PoissonSampler",
    "TruncatedPoissonSampler",
    "CyclicPoissonSampler",
    "PartitionType",
    "distributed",
]
