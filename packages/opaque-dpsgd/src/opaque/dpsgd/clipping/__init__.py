"""DP-SGD clipping: fixed, AUTO-S, per-group, and adaptive thresholding.

Headline factories:

- :func:`clipped_grad` — fixed-threshold per-example clipping
- :func:`auto_clipped_grad` — AUTO-S (Bu et al., NeurIPS 2023)
- :func:`per_group` — build :class:`~opaque.types.PerGroup` groupings
- :func:`adaptive_clipped_grad` — adaptive clipping (Andrew et al., 2021)

Fixed and AUTO-S are implemented in :mod:`opaque._clipping` and re-exported
here as the **canonical** import path for DP-SGD training code.  Adaptive
clipping is DP-SGD-only: its threshold drifts across steps, which violates
the constant-sensitivity assumption matrix-factorization proofs require.

State and auxiliary dataclasses live in :mod:`opaque.dpsgd.clipping.types`.
AUTO-S function-level helpers live in :mod:`opaque.dpsgd.clipping.fun`.
"""

from opaque._clipping import auto_clipped_grad, clipped_grad, per_group
from opaque.dpsgd.clipping._adaptive import adaptive_clipped_grad

import opaque.dpsgd.clipping._distributed  # noqa: F401  (registers sync handlers)

__all__ = [
    "adaptive_clipped_grad",
    "auto_clipped_grad",
    "clipped_grad",
    "per_group",
]
