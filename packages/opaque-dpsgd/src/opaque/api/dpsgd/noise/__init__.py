"""DP-SGD noise mechanisms impl — Gaussian (optionally bounded)."""

import opaque.api.dpsgd.noise._distributed  # noqa: F401  (registers sync handlers)
from opaque.api.dpsgd.noise._gaussian import gaussian_noise
from opaque.api.engine.noise_allocation import noise_stddev

__all__ = ["gaussian_noise", "noise_stddev"]
