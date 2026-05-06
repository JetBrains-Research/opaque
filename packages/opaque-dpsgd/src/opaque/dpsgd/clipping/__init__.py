"""DP-SGD-specific clipping mechanisms: adaptive and AUTO-S.

Headline factories:

- :func:`adaptive_clipped_grad` — adaptive gradient clipping (Andrew et al. 2021)
- :func:`auto_clipped_grad` — AUTO-S smooth-scaled gradient clipping (Bu et al. 2023)

Power-user :func:`auto_clipped_fun` (function-level AUTO-S) lives in
:mod:`opaque.dpsgd.clipping.fun`.

State and auxiliary dataclasses (``AdaptiveClipState``,
``AdaptiveClippedGradAux``, ``AutoClipState``, ``AutoClippedFunAux``,
``AutoClippedGradAux``) live in :mod:`opaque.dpsgd.clipping.types`.
"""

from opaque.dpsgd.clipping._adaptive import adaptive_clipped_grad
from opaque.dpsgd.clipping._auto import auto_clipped_grad

import opaque.dpsgd.clipping._distributed  # noqa: F401  (registers sync handlers)

__all__ = [
    "adaptive_clipped_grad",
    "auto_clipped_grad",
]
