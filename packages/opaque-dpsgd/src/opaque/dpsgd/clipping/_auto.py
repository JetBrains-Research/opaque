"""AUTO-S clipping symbols (see :mod:`opaque.dpsgd.clipping`)."""

from __future__ import annotations

from opaque._clipping import auto_clipped_grad
from opaque._clipping.fun import auto_clipped_fun
from opaque._clipping.types import (
    AutoClippedFunAux,
    AutoClippedGradAux,
    AutoClipState,
)

__all__ = [
    "AutoClipState",
    "AutoClippedFunAux",
    "AutoClippedGradAux",
    "auto_clipped_fun",
    "auto_clipped_grad",
]
