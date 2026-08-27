"""Precision types — the loss-scaler bundle and its state.

:func:`opaque.precision.loss_scaler` builds both; they live here for
``isinstance`` checks and type annotations, matching
:mod:`opaque.optimizers.types`.
"""

from opaque.api.engine.precision import LossScaler, LossScalerState

__all__ = ["LossScaler", "LossScalerState"]
