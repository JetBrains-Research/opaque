"""Public type definitions for :mod:`opaque.optimizers`.

Re-exports the per-optimizer state dataclasses for type annotations. The
functional factories (``adam``, ``adamw``, ``lion``, …) live in
:mod:`opaque.optimizers`.  Checkpointing uses :mod:`opaque.serialization`
(``state_dict`` / ``from_state_dict``).
"""

from __future__ import annotations

from opaque.api.optimizers._adadelta import AdadeltaState
from opaque.api.optimizers._adafactor import AdafactorState
from opaque.api.optimizers._adagrad import AdagradState
from opaque.api.optimizers._adam import AdamState
from opaque.api.optimizers._ademamix import AdEMAMixState
from opaque.api.optimizers._lion import LionState
from opaque.api.optimizers._radam import RAdamState
from opaque.api.optimizers._rmsprop import RMSpropState
from opaque.api.optimizers._schedule_free import ScheduleFreeState

__all__ = [
    "AdEMAMixState",
    "AdadeltaState",
    "AdafactorState",
    "AdagradState",
    "AdamState",
    "LionState",
    "RAdamState",
    "RMSpropState",
    "ScheduleFreeState",
]
