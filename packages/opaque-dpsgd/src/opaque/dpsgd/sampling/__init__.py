"""DP-SGD-specific samplers.

Exposes the generic :class:`PoissonSampler` (variable-size batches, the
default for DP-SGD training) and :class:`TruncatedPoissonSampler` (fixed-size
truncation variant).
"""

from opaque.dpsgd.sampling.poisson import PoissonSampler
from opaque.dpsgd.sampling.truncated_poisson import TruncatedPoissonSampler

__all__ = ["PoissonSampler", "TruncatedPoissonSampler"]
