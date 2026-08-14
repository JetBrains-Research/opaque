"""Backend-neutral optimizer factories.

All factories follow the explicit-state pattern used by the rest of
Opaque: a factory takes the parameter pytree and hyper-parameters, and
returns ``(step_fn, state)``.  The returned ``step_fn`` produces signed
updates; callers apply them with :func:`apply_updates`.
"""

from __future__ import annotations

from opaque.api.engine.optimizers._adadelta import adadelta
from opaque.api.engine.optimizers._adafactor import adafactor
from opaque.api.engine.optimizers._adagrad import adagrad
from opaque.api.engine.optimizers._adam import adam, adamw
from opaque.api.engine.optimizers._ademamix import ademamix
from opaque.api.engine.optimizers._chain import apply_updates
from opaque.api.engine.optimizers._lion import lion
from opaque.api.engine.optimizers._radam import radam
from opaque.api.engine.optimizers._rmsprop import rmsprop
from opaque.api.engine.optimizers._schedule_free import schedule_free
from opaque.api.engine.optimizers._sgd import sgd

__all__ = [
    "adam",
    "adamw",
    "adadelta",
    "adafactor",
    "adagrad",
    "ademamix",
    "apply_updates",
    "lion",
    "radam",
    "rmsprop",
    "schedule_free",
    "sgd",
]
