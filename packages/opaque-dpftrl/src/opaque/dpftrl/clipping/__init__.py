"""DP-FTRL clipping: fixed threshold, AUTO-S, and per-group norms.

Training scripts should import from here. Adaptive thresholding is only
available under :mod:`opaque.dpsgd.clipping` (DP-SGD).
"""

from opaque.api.engine.clipping import auto_clipped_grad, clipped_grad, per_group

import opaque.api.engine.clipping._distributed  # noqa: F401  (registers fixed/AUTO-S sync)

__all__ = [
    "auto_clipped_grad",
    "clipped_grad",
    "per_group",
]
