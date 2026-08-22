"""DP-SGD noise mechanisms façade — Gaussian (optionally bounded).

State (``GaussianNoiseState``) lives in :mod:`opaque.dpsgd.noise.types`.
"""

from opaque.api.dpsgd.noise import gaussian_noise
from opaque.dpsgd.noise import types

__all__ = ["gaussian_noise", "types"]
