"""Type definitions for clipping operations."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass


class ClipState(ABC):
    """Base class for clipping state.

    Clipping state is the explicit state token returned by clipping transforms.
    Fixed clipping uses it as an immutable marker; adaptive schemes may carry
    the threshold and counters needed for the next step. Privacy calibration
    metadata lives on the returned :class:`opaque.bounded.BoundedPytree`, not
    on the state object.

    Example:
        >>> from opaque.clipping import clipped_grad
        >>> loss_fn = lambda params, x, y: ((x @ params - y) ** 2).mean()
        >>>
        >>> # Fixed clipping returns (callable, ClipState)
        >>> grad_fn, clip_state = clipped_grad(loss_fn, clipping_norm=1.0, batch_argnums=(1, 2))
        >>> grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
        >>>
        >>> # Noise calibration reads the bound metadata from clipped outputs
        >>> from opaque.dpsgd.noise import gaussian_noise
        >>> noise_fn, noise_state = gaussian_noise(noise_multiplier=1.1, key=key(0))
        >>> noisy_grads, noise_state = noise_fn(grads, noise_state)
    """


@dataclass(frozen=True)
class FixedClipState(ClipState):
    """Marker state for fixed (non-adaptive) clipping."""


__all__ = [
    "ClipState",
    "FixedClipState",
]
