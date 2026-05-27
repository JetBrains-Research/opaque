"""Precision primitives — functional loss scaler for fp16 training.

Mirrors :class:`torch.amp.GradScaler` against the pytree gradient flow
used by Opaque's DP step. See :mod:`opaque.api.engine.precision` for
implementation notes and the DP-critical ordering invariant.
"""

from opaque.api.engine.precision import (
    LossScaler,
    LossScalerState,
    all_finite,
    loss_scaler,
)

__all__ = [
    "LossScaler",
    "LossScalerState",
    "all_finite",
    "loss_scaler",
]
