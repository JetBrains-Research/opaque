"""Per-example gradient clipping for differential privacy.

This module provides utilities for clipping gradients and function outputs
to bounded L2 sensitivity, a key requirement for DP-SGD.
"""

from opaque.clipping.adaptive import (
    AdaptiveClippedGradAux,
    AdaptiveClipState,
    adaptive_clipped_grad,
)
from opaque.clipping.clipped_fun import ClippedFunAux, clipped_fun
from opaque.clipping.clipped_grad import ClippedGradAux, clipped_grad
from opaque.clipping import distributed
from opaque.clipping.distributed import (sync_adaptive_clip_state,
                                         sync_adaptive_clipped_grad_aux,
                                         sync_clip_state)
from opaque.clipping.pytree import clip_pytree
from opaque.clipping.types import ClipState, FixedClipState, NeighboringRelation

__all__ = [
    # Core clipping functions
    "clip_pytree",
    "clipped_fun",
    "clipped_grad",
    "adaptive_clipped_grad",
    # State types
    "ClipState",
    "FixedClipState",
    "AdaptiveClipState",
    # Auxiliary outputs
    "ClippedFunAux",
    "ClippedGradAux",
    "AdaptiveClippedGradAux",
    # Synchronization helpers
    "sync_clip_state",
    "sync_adaptive_clip_state",
    "sync_adaptive_clipped_grad_aux",
    # Types
    "NeighboringRelation",
    "distributed",
]
