"""DP-SGD noise mechanisms façade — Gaussian and truncated Gaussian.

State (``GaussianNoiseState``) lives in :mod:`opaque.dpsgd.noise.types`.
"""

from opaque.api.dpsgd.noise import gaussian_noise, truncated_gaussian_noise

__all__ = ["gaussian_noise", "truncated_gaussian_noise"]
