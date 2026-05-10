"""Backward-compat shim: AUTO-S clipping moved to :mod:`opaque.clipping`.

AUTO-S has a constant, data-independent per-record sensitivity bound and is
therefore algorithm-agnostic — it composes with both DP-SGD's Gaussian
mechanism and DP-FTRL's matrix-factorization mechanisms.  The canonical
home is now :mod:`opaque.clipping`; this module re-exports the same names
so existing ``from opaque.dpsgd.clipping import auto_clipped_grad`` imports
keep working.
"""

from __future__ import annotations

from opaque.clipping import auto_clipped_grad
from opaque.clipping.fun import auto_clipped_fun
from opaque.clipping.types import (
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
