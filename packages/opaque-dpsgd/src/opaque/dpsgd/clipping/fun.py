"""Backward-compat shim: AUTO-S function-level scaling moved to
:mod:`opaque.clipping.fun`.

This submodule used to host :func:`auto_clipped_fun`; the canonical home
is now :mod:`opaque.clipping.fun` (AUTO-S is algorithm-agnostic — its
per-record sensitivity bound is constant, so it composes with both
DP-SGD's Gaussian mechanism and DP-FTRL's matrix-factorization
mechanisms).  This module re-exports :func:`auto_clipped_fun` so existing
``from opaque.dpsgd.clipping.fun import auto_clipped_fun`` imports keep
working.
"""

from __future__ import annotations

from opaque.clipping._auto import auto_clipped_fun

__all__ = ["auto_clipped_fun"]
