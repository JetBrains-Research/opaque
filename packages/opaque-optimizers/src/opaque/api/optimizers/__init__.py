"""Backend-neutral optimizer factories.

All factories follow the explicit-state pattern used by the rest of
Opaque: a factory takes the parameter pytree and hyper-parameters, and
returns ``(step_fn, state)``.  The returned ``step_fn`` produces signed
updates; callers apply them with :func:`apply_updates`.
"""

from __future__ import annotations

# Import-time side effect: optimizer states become discoverable by
# ``opaque.distributed.sync()``. Registration belongs here rather than in a
# provider — auditing optimizer state is optimizer knowledge, and a state
# can only exist once this package has been imported.
from opaque.api.optimizers import _distributed as _distributed
from opaque.api.optimizers._adadelta import adadelta
from opaque.api.optimizers._adafactor import adafactor
from opaque.api.optimizers._adagrad import adagrad
from opaque.api.optimizers._adam import adam, adamw
from opaque.api.optimizers._ademamix import ademamix
from opaque.api.optimizers._chain import apply_updates
from opaque.api.optimizers._lion import lion
from opaque.api.optimizers._radam import radam
from opaque.api.optimizers._rmsprop import rmsprop
from opaque.api.optimizers._schedule_free import schedule_free
from opaque.api.optimizers._sgd import sgd

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
