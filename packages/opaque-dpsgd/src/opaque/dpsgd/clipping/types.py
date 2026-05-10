"""Public type definitions for :mod:`opaque.dpsgd.clipping`.

Re-exports the adaptive clipping state and auxiliary dataclasses for type
annotations, plus AUTO-S types from :mod:`opaque._clipping.types`.
"""

from __future__ import annotations

from opaque._clipping.types import (
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
