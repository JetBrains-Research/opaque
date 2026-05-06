"""Public type definitions for :mod:`opaque.optimizers`.

Re-exports the per-optimizer state dataclasses for type annotations. The
functional factories (``adam``, ``adamw``, ``lion``, …) live in
:mod:`opaque.optimizers`.  Checkpointing uses :mod:`opaque.serialization`
(``state_dict`` / ``from_state_dict``).
"""

from __future__ import annotations

from opaque.optimizers._adadelta import AdadeltaState
from opaque.optimizers._adafactor import AdafactorState
from opaque.optimizers._adagrad import AdagradState
from opaque.optimizers._adam import AdamState
from opaque.optimizers._ademamix import AdEMAMixState
from opaque.optimizers._lion import LionState
from opaque.optimizers._radam import RAdamState
from opaque.optimizers._rmsprop import RMSpropState
from opaque.optimizers._schedule_free import ScheduleFreeState

__all__ = [
    "AdamState",
    "AdadeltaState",
    "AdafactorState",
    "AdagradState",
    "AdEMAMixState",
    "LionState",
    "RAdamState",
    "RMSpropState",
    "ScheduleFreeState",
]
