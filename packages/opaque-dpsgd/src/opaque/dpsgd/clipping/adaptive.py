"""Public surface for adaptive clipping state (checkpointing, tests).

Implementation lives in :mod:`opaque.dpsgd.clipping._adaptive`; this module
exists so imports like ``opaque.dpsgd.clipping.adaptive`` resolve without
reaching into private modules.
"""

from __future__ import annotations

from opaque.dpsgd.clipping._adaptive import (
    AdaptiveClippedGradAux,
    AdaptiveClipState,
    adaptive_clipped_grad,
)

__all__ = [
    "AdaptiveClipState",
    "AdaptiveClippedGradAux",
    "adaptive_clipped_grad",
]
