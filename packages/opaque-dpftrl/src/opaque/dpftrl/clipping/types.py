"""Type aliases for DP-FTRL clipping (AUTO-S + fixed-clip state)."""

from __future__ import annotations

from opaque._clipping.types import (
    AutoClippedFunAux,
    AutoClippedGradAux,
    AutoClipState,
    ClippedFunAux,
    ClippedGradAux,
    ClipPytreeAux,
    FixedClipState,
)

__all__ = [
    "AutoClipState",
    "AutoClippedFunAux",
    "AutoClippedGradAux",
    "ClipPytreeAux",
    "ClippedFunAux",
    "ClippedGradAux",
    "FixedClipState",
]
