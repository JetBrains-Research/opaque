"""Post-processing denoisers for noisy gradients (e.g. DiSK)."""

from opaque.denoising.disk import DiskDenoiserState, disk_denoiser
from opaque.denoising.types import DenoiserState

import opaque.denoising._distributed  # noqa: F401  (registers sync handlers)

__all__ = [
    "DenoiserState",
    "DiskDenoiserState",
    "disk_denoiser",
]
