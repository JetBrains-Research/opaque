"""Post-processing denoisers for noisy gradients (e.g. DiSK)."""

from opaque.denoising._disk import disk_denoiser

import opaque.denoising._distributed  # noqa: F401  (registers sync handlers)

__all__ = [
    "disk_denoiser",
]
