"""DP-SGD-specific samplers.

Exposes :class:`PoissonSampler`, which covers both plain Poisson
subsampling (default) and the truncated-Poisson variant via the
``truncated_batch_size`` keyword argument.
"""

from opaque.dpsgd.sampling._poisson import PoissonSampler

__all__ = ["PoissonSampler"]
