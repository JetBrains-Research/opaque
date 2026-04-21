"""Per-example gradient clipping for differential privacy.

This module provides utilities for clipping gradients and function outputs
to bounded L2 sensitivity, a key requirement for DP-SGD.
"""

from opaque.core.clipping import distributed
from opaque.core.clipping.adaptive import (
    AdaptiveClippedGradAux,
    AdaptiveClipState,
    adaptive_clipped_grad,
)
from opaque.core.clipping.auto import (
    AutoClippedFunAux,
    AutoClippedGradAux,
    AutoClipState,
    auto_clipped_fun,
    auto_clipped_grad,
)
from opaque.core.clipping.clipped_fun import ClippedFunAux, clipped_fun
from opaque.core.clipping.clipped_grad import ClippedGradAux, clipped_grad
from opaque.core.clipping.distributed import (
    sync_adaptive_clip_state,
    sync_aux,
    sync_clip_state,
)
from opaque.core.clipping.pytree import auto_scale_pytree, clip_pytree
from opaque.core.clipping.types import ClipState, FixedClipState

__all__ = [
    # Core clipping functions
    "clip_pytree",
    "auto_scale_pytree",
    "clipped_fun",
    "clipped_grad",
    "adaptive_clipped_grad",
    "auto_clipped_fun",
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
    "AutoClippedFunAux",
    "AutoClippedGradAux",
    # Synchronization helpers
    "sync_clip_state",
    "sync_adaptive_clip_state",
    "sync_aux",
    "distributed",
]
