"""Type definitions for gradient denoising (post-processing on noisy gradients)."""

from __future__ import annotations

from abc import ABC


class DenoiserState(ABC):
    """Base class for denoiser state.

    All denoisers return ``(denoise_fn, state)`` where ``state`` inherits from
    this class, mirroring :class:`~opaque.noise.types.NoiseState` and
    :class:`~opaque.clipping.types.ClipState`.

    Attributes:
        _step_counter: Number of ``denoise_fn`` calls completed.
    """

    _step_counter: int
    """Number of denoise_fn calls completed."""
