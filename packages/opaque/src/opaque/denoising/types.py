"""Type definitions for gradient denoising (post-processing on noisy gradients)."""

from __future__ import annotations

import dataclasses
from abc import ABC
from typing import Any


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


@dataclasses.dataclass(frozen=True)
class DiskDenoiserState(DenoiserState):
    """Immutable state for :func:`~opaque.denoising.disk_denoiser` (DiSK)."""

    _estimate: Any
    _error_var: Any
    _step_counter: int
