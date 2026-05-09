"""DP-SGD-specific clipping: adaptive thresholding.

Headline factories:

- :func:`adaptive_clipped_grad` — adaptive gradient clipping (Andrew et al. 2021)

The adaptive threshold drifts across steps based on the noisy clipping
rate, which violates the constant per-step sensitivity assumption that
matrix-factorization privacy proofs rely on; adaptive clipping is therefore
exclusive to DP-SGD.

AUTO-S clipping (:func:`auto_clipped_grad`, Bu et al. 2023) lives in
:mod:`opaque.clipping` because its per-record sensitivity bound is
constant and data-independent; that makes it interchangeable with fixed
clipping under both DP-SGD's Gaussian mechanism and DP-FTRL's
matrix-factorization mechanisms.  :func:`auto_clipped_grad`,
:func:`opaque.clipping.fun.auto_clipped_fun`, and the AUTO-S state / aux
dataclasses are re-exported here for backward compatibility.

State and auxiliary dataclasses (``AdaptiveClipState``,
``AdaptiveClippedGradAux``, plus the re-exported ``AutoClipState``,
``AutoClippedFunAux``, ``AutoClippedGradAux``) live in
:mod:`opaque.dpsgd.clipping.types`.
"""

from opaque.clipping._auto import auto_clipped_grad
from opaque.dpsgd.clipping._adaptive import adaptive_clipped_grad

import opaque.dpsgd.clipping._distributed  # noqa: F401  (registers sync handlers)

__all__ = [
    "adaptive_clipped_grad",
    "auto_clipped_grad",
]
