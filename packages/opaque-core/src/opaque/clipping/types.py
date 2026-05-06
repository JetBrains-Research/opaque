"""Public type definitions for :mod:`opaque.clipping`.

Re-exports the clipping-specific state and auxiliary dataclasses for
type annotations. The cross-cutting DP types (``ClipState`` base,
``ClippedPytree``, ``PerGroup``, ``MaxNorm``, ``clipped()`` factory)
live in :mod:`opaque.types`.
"""

from __future__ import annotations

from opaque.clipping._clipped_fun import ClippedFunAux
from opaque.clipping._clipped_fun import FixedClipState
from opaque.clipping._clipped_grad import ClippedGradAux
from opaque.clipping._pytree import ClipPytreeAux

__all__ = [
    "FixedClipState",
    "ClippedGradAux",
    "ClippedFunAux",
    "ClipPytreeAux",
]
