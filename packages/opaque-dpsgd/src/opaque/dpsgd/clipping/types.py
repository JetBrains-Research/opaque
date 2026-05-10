"""Clipping state and auxiliary types for :mod:`opaque.dpsgd.clipping`.

Includes adaptive (DP-SGD) and AUTO-S dataclasses used in type annotations.
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
