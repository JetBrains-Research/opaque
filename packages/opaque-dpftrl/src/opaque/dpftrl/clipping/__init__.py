"""DP-FTRL clipping façade — fixed threshold, AUTO-S, and per-group norms.

Adaptive thresholding is only available under :mod:`opaque.dpsgd.clipping`
(DP-SGD). State and aux types live in :mod:`opaque.dpftrl.clipping.types`.
"""

from opaque.api.engine.clipping import auto_clipped_grad, clipped_grad, per_group

__all__ = [
    "auto_clipped_grad",
    "clipped_grad",
    "per_group",
]
