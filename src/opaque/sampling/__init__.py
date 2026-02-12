"""Sampling strategies for DP-SGD training.

This module provides PyTorch-compatible samplers for differential privacy,
including Poisson sampling for privacy amplification.
"""

from opaque.sampling.poisson import PoissonSampler
from opaque.sampling.truncated_poisson import TruncatedPoissonSampler

__all__ = [
    "PoissonSampler",
    "TruncatedPoissonSampler",
]
