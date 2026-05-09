"""Public type definitions for :mod:`opaque.dpsgd.clipping`.

Re-exports the adaptive clipping state and auxiliary dataclasses for type
annotations.  AUTO-S types now live in :mod:`opaque.clipping.types`; they
are re-exported here for backward compatibility.
"""

from __future__ import annotations

from opaque.clipping._auto import (
    AutoClippedFunAux,
    AutoClippedGradAux,
    AutoClipState,
)
from opaque.dpsgd.clipping._adaptive import AdaptiveClippedGradAux, AdaptiveClipState

__all__ = [
    "AdaptiveClipState",
    "AdaptiveClippedGradAux",
    "AutoClipState",
    "AutoClippedFunAux",
    "AutoClippedGradAux",
]
