"""Public type definitions for :mod:`opaque.api.engine.scheduling`.

Pure re-export façade — the :data:`Schedule` callable type alias and
the recipe dataclasses that implement it both live in their
implementation modules; this module just gathers them for ``isinstance``
checks and type annotations, matching how
:mod:`opaque.dpftrl.noise.types`, :mod:`opaque.dpsgd.noise.types`, etc.
are structured.
"""

from __future__ import annotations

from opaque.api.engine.scheduling._constant import ConstantSchedule
from opaque.api.engine.scheduling._cosine import CosineSchedule
from opaque.api.engine.scheduling._exponential import ExponentialSchedule
from opaque.api.engine.scheduling._inverse_sqrt import InverseSqrtSchedule
from opaque.api.engine.scheduling._linear import LinearSchedule
from opaque.api.engine.scheduling._one_minus_sqrt import OneMinusSqrtSchedule
from opaque.api.engine.scheduling._polynomial import PolynomialSchedule
from opaque.api.engine.scheduling._schedule import Schedule
from opaque.api.engine.scheduling._warmup_stable_decay import WarmupStableDecay
from opaque.api.engine.scheduling._with_restarts import WithRestarts
from opaque.api.engine.scheduling._with_warmup import WithWarmup

__all__ = [
    "Schedule",
    "ConstantSchedule",
    "LinearSchedule",
    "PolynomialSchedule",
    "ExponentialSchedule",
    "CosineSchedule",
    "InverseSqrtSchedule",
    "OneMinusSqrtSchedule",
    "WithWarmup",
    "WithRestarts",
    "WarmupStableDecay",
]
