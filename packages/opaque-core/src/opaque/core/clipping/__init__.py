"""Per-example gradient clipping primitives (algorithm-agnostic).

Core exposes the generic clipping building blocks (pytree clip, fixed
``clipped_fun`` / ``clipped_grad``, the base :class:`ClipState`, and
distributed sync helpers for those). DP-SGD-specific variants
(``adaptive_*``, ``auto_*``) live in :mod:`opaque.dpsgd.clipping`.
"""

from opaque.core.clipping import distributed
from opaque.core.clipping.clipped_fun import ClippedFunAux, clipped_fun
from opaque.core.clipping.clipped_grad import ClippedGradAux, clipped_grad
from opaque.core.clipping.distributed import sync_aux, sync_clip_state
from opaque.core.clipping.pytree import auto_scale_pytree, clip_pytree
from opaque.core.clipping.types import ClipState, FixedClipState

__all__ = [
    "clip_pytree",
    "auto_scale_pytree",
    "clipped_fun",
    "clipped_grad",
    "ClipState",
    "FixedClipState",
    "ClippedFunAux",
    "ClippedGradAux",
    "sync_clip_state",
    "sync_aux",
    "distributed",
]
