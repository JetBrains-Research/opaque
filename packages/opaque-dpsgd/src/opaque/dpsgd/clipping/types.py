"""Public type definitions for :mod:`opaque.dpsgd.clipping`.

Re-exports the adaptive- and AUTO-S clipping state and auxiliary
dataclasses for type annotations.
"""

from __future__ import annotations

from opaque.dpsgd.clipping._adaptive import AdaptiveClippedGradAux, AdaptiveClipState
from opaque.dpsgd.clipping._auto import (
    AutoClippedFunAux,
    AutoClippedGradAux,
    AutoClipState,
)

__all__ = [
    "AdaptiveClipState",
    "AdaptiveClippedGradAux",
    "AutoClipState",
    "AutoClippedFunAux",
    "AutoClippedGradAux",
]
