"""Noise generation for differential privacy."""

from opaque.noise import distributed
from opaque.noise.band_mf_noise import band_mf_noise
from opaque.noise.blt_mf_noise import blt_mf_noise
from opaque.noise.bounded_gaussian_noise import bounded_gaussian_noise
from opaque.noise.custom_mf_noise import custom_mf_noise
from opaque.noise.dense_mf_noise import dense_mf_noise
from opaque.noise.gaussian_noise import gaussian_noise
from opaque.noise.identity_mf_noise import identity_mf_noise

__all__ = [
    "band_mf_noise",
    "blt_mf_noise",
    "bounded_gaussian_noise",
    "custom_mf_noise",
    "dense_mf_noise",
    "gaussian_noise",
    # Backwards-compatible convenience alias
    "gaussian",
    "identity_mf_noise",
    "distributed",
]

# Convenience alias for callers using the short `gaussian` name in docs/examples
gaussian = gaussian_noise
