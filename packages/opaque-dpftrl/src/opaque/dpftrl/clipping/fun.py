"""AUTO-S function-level scaling (re-export for DP-FTRL)."""

from __future__ import annotations

from opaque._clipping.fun import auto_clipped_fun, auto_scale_pytree, clip_pytree

__all__ = ["auto_clipped_fun", "auto_scale_pytree", "clip_pytree"]
