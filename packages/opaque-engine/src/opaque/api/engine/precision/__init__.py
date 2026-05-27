"""Precision primitives for the functional training step.

Currently exposes a functional loss scaler (the analog of
:class:`torch.amp.GradScaler` for the pytree-shaped gradient path); future
additions (e.g. master-weight orchestration) would live alongside it.
"""

from opaque.api.engine.precision._loss_scaler import (
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
