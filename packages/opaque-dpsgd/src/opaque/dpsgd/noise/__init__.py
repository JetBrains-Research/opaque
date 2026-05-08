"""DP-SGD noise mechanisms: Gaussian, truncated Gaussian, per-group.

The Gaussian noise state (``GaussianNoiseState``) lives in
:mod:`opaque.dpsgd.noise.types`.
"""

from opaque.dpsgd.noise._gaussian import gaussian_noise
from opaque.dpsgd.noise._per_group_noise import (
    paired_noise_stddevs,
    per_group_noise_stddev,
)
from opaque.dpsgd.noise._truncated_gaussian import truncated_gaussian_noise

import opaque.dpsgd.noise._distributed  # noqa: F401  (registers sync handlers)

__all__ = [
    "gaussian_noise",
    "truncated_gaussian_noise",
    "per_group_noise_stddev",
    "paired_noise_stddevs",
]
