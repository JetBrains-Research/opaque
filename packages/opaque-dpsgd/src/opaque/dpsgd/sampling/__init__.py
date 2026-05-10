"""DP-SGD-specific samplers.

Exposes :class:`PoissonSubsampler`, which covers both plain Poisson
subsampling (default) and the truncated-Poisson variant via the
``truncated_batch_size`` keyword argument.
"""

from opaque.dpsgd.sampling._poisson import PoissonSubsampler

__all__ = ["PoissonSubsampler"]
