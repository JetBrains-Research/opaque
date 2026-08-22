"""Precision primitives — functional loss scaler for fp16 training.

Mirrors :class:`torch.amp.GradScaler` against the pytree gradient flow
used by Opaque's DP step. See :mod:`opaque.api.engine.precision` for
implementation notes and the DP-critical ordering invariant.

:func:`loss_scaler` is the entry point; the ``LossScaler`` bundle and
``LossScalerState`` it returns live in :mod:`opaque.precision.types`.
"""

from opaque.api.engine.precision import all_finite, loss_scaler
from opaque.precision import types

__all__ = [
    "all_finite",
    "loss_scaler",
    "types",
]
