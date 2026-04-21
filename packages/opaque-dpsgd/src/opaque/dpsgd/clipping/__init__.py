"""DP-SGD-specific clipping mechanisms: adaptive and AUTO-S."""

from opaque.dpsgd.clipping import distributed
from opaque.dpsgd.clipping.adaptive import (
    AdaptiveClippedGradAux,
    AdaptiveClipState,
    adaptive_clipped_grad,
)
from opaque.dpsgd.clipping.auto import (
    AutoClippedFunAux,
    AutoClippedGradAux,
    AutoClipState,
    auto_clipped_fun,
    auto_clipped_grad,
)
from opaque.dpsgd.clipping.distributed import (
    sync_adaptive_clip_state,
    sync_adaptive_clipped_grad_aux,
)

__all__ = [
    "adaptive_clipped_grad",
    "AdaptiveClipState",
    "AdaptiveClippedGradAux",
    "auto_clipped_fun",
    "auto_clipped_grad",
    "AutoClipState",
    "AutoClippedFunAux",
    "AutoClippedGradAux",
    "sync_adaptive_clip_state",
    "sync_adaptive_clipped_grad_aux",
    "distributed",
]
