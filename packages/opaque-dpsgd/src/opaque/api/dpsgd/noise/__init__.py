"""DP-SGD noise mechanisms impl — Gaussian (optionally bounded)."""

from opaque.api.dpsgd.noise._gaussian import gaussian_noise

import opaque.api.dpsgd.noise._distributed  # noqa: F401  (registers sync handlers)

__all__ = ["gaussian_noise"]
