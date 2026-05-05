"""Concrete fixed-clipping marker state.

The cross-cutting ``ClipState`` base, the ``ClippedPytree`` wrapper,
the ``MaxNorm`` alias, and the ``clipped()`` factory live in
:mod:`opaque.types` (where they sit alongside ``NoiseState``,
``NoisedPytree`` and the paired-stream outputs).  This module keeps
the concrete fixed-clipping marker that pairs with
:func:`opaque.clipping.clipped_grad`.
"""

from __future__ import annotations

from dataclasses import dataclass

# Imported privately so ``from opaque.clipping.types import ClipState``
# does not work — :mod:`opaque.types` is the single canonical home.
from opaque.types import ClipState as _ClipState


@dataclass(frozen=True)
class FixedClipState(_ClipState):
    """Marker state for fixed (non-adaptive) clipping.

    Carries no fields; the configured clipping threshold flows through
    the ``ClippedPytree.max_norm`` returned by every clipping op, not
    through the state token.

    Example:
        >>> from opaque.clipping import clipped_grad
        >>> loss_fn = lambda params, x, y: ((x @ params - y) ** 2).mean()
        >>>
        >>> # Fixed clipping returns (callable, FixedClipState)
        >>> grad_fn, clip_state = clipped_grad(loss_fn, clipping_norm=1.0, batch_argnums=(1, 2))
        >>> grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
        >>>
        >>> # Noise calibration reads ``max_norm`` from the clipped output
        >>> from opaque.dpsgd.noise import gaussian_noise
        >>> noise_fn, noise_state = gaussian_noise(noise_multiplier=1.1, key=key(0))
        >>> noisy_grads, noise_state = noise_fn(grads, noise_state)
    """


__all__ = ["FixedClipState"]
