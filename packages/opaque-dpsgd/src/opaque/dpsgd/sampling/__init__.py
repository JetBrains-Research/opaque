"""DP-SGD-specific samplers.

Exposes :class:`PoissonSubsampler`, which covers both plain Poisson
subsampling (default).  Pass ``truncated_batch_size`` to cap batch size
(truncated Poisson; weaker privacy than plain Poisson at the same rate unless
noise is recalibrated—use matching accounting).
"""

from opaque.dpsgd.sampling._poisson import PoissonSubsampler

__all__ = ["PoissonSubsampler"]
