"""DP-FTRL clipping: fixed threshold and AUTO-S (MF-safe).

:func:`adaptive_clipped_grad` from DP-SGD is **not** available here — adaptive
thresholds violate the constant per-step sensitivity assumption used in
matrix-factorization privacy proofs.

Implementation is shared via :mod:`opaque._clipping`; this module is the
canonical import path for DP-FTRL application code.
"""

from opaque._clipping import auto_clipped_grad, clipped_grad, per_group

import opaque._clipping._distributed  # noqa: F401  (registers fixed/AUTO-S sync)

__all__ = [
    "auto_clipped_grad",
    "clipped_grad",
    "per_group",
]
