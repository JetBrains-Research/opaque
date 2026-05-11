"""DP-SGD samplers façade — Poisson subsampling.

Exposes :class:`PoissonSampler`, which covers both plain Poisson
subsampling (default). Pass ``truncated_batch_size`` to cap batch size
(truncated Poisson; weaker privacy than plain Poisson at the same rate
unless noise is recalibrated — use matching accounting).
"""

from opaque.api.dpsgd.sampling import PoissonSampler

__all__ = ["PoissonSampler"]
