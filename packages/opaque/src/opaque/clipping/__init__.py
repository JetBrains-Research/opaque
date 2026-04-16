"""Per-example gradient clipping for differential privacy.

This module provides utilities for clipping gradients and function outputs
to bounded L2 sensitivity, a key requirement for DP-SGD.
"""

from opaque.clipping import distributed
from opaque.clipping.adaptive import (
    AdaptiveClippedGradAux,
    AdaptiveClipState,
    adaptive_clipped_grad,
)
from opaque.clipping.auto import (
    AutoClippedGradAux,
    AutoClipState,
    auto_clipped_grad,
)
from opaque.clipping.clipped_fun import ClippedFunAux, clipped_fun
from opaque.clipping.clipped_grad import ClippedGradAux, clipped_grad
from opaque.clipping.distributed import (
    sync_adaptive_clip_state,
    sync_aux,
    sync_clip_state,
)
from opaque.clipping.pytree import clip_pytree
from opaque.clipping.types import ClipState, FixedClipState

__all__ = [
    # Core clipping functions
    "clip_pytree",
    "clipped_fun",
    "clipped_grad",
    "adaptive_clipped_grad",
    "auto_clipped_grad",
    # State types
    "ClipState",
    "FixedClipState",
    "AdaptiveClipState",
    "AutoClipState",
    # Auxiliary outputs
    "ClippedFunAux",
    "ClippedGradAux",
    "AdaptiveClippedGradAux",
    "AutoClippedGradAux",
    # Synchronization helpers
    "sync_clip_state",
    "sync_adaptive_clip_state",
    "sync_aux",
    "distributed",
]
