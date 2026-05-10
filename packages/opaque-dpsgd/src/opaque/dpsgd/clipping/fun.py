"""AUTO-S function-level scaling (re-export).

Implementation lives in :mod:`opaque._clipping.fun`.  Import from here for
DP-SGD tutorials; :mod:`opaque.dpftrl.clipping.fun` mirrors this for FTRL.
"""

from __future__ import annotations

from opaque._clipping.fun import auto_clipped_fun

__all__ = ["auto_clipped_fun"]
