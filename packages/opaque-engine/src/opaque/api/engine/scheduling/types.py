"""Public type definitions for :mod:`opaque.api.engine.scheduling`.

Pure re-export façade — the :data:`Schedule` callable type alias and
the recipe dataclasses that implement it both live in their
implementation modules; this module just gathers them for ``isinstance``
checks and type annotations, matching how
:mod:`opaque.dpftrl.noise.types`, :mod:`opaque.dpsgd.noise.types`, etc.
are structured.
"""

from __future__ import annotations

from opaque.api.engine.scheduling._compose import (
    WarmupStableDecay,
    WithRestarts,
    WithWarmup,
)
from opaque.api.engine.scheduling._curves import (
    ConstantSchedule,
    CosineSchedule,
    ExponentialSchedule,
    InverseSqrtSchedule,
    LinearSchedule,
    OneMinusSqrtSchedule,
    PolynomialSchedule,
    Schedule,
)

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
