"""DiSK gradient denoising — Zhang et al., ICLR 2025.

Public entry point; implementation lives in :mod:`opaque.denoising._kalman`.
"""

from __future__ import annotations

from opaque.denoising._kalman import DiskDenoiserState, disk_denoiser

__all__ = [
    "DiskDenoiserState",
    "disk_denoiser",
]
