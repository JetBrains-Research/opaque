"""Composition primitives for step-indexed schedules.

Like :mod:`._curves`, each wrapper is a ``@dataclass(frozen=True)``
carrying an inner schedule + composition args; ``__call__`` evaluates
the composed schedule.  The factory functions ``with_warmup``,
``with_restarts``, ``warmup_stable_decay`` are kept for backward
compatibility and just construct the matching dataclass.

Named ramps and decays (``"linear"``, ``"cosine"``, ``"1-sqrt"``) are
string discriminators — the dataclass dispatches in ``__call__`` so
recipes round-trip through serialization without carrying any closure.
A user-supplied callable still works; the resulting instance is a
valid Schedule but won't round-trip through the universal serializer.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Union

from opaque.api.engine.scheduling._curves import ConstantSchedule
from opaque.api.engine.scheduling.types import Schedule

__all__ = [
    "WithWarmup",
    "WithRestarts",
    "WarmupStableDecay",
    "with_warmup",
    "with_restarts",
    "warmup_stable_decay",
]


# Named ramps (`progress in [0, 1] -> factor in [0, 1]`).
_NAMED_RAMPS: dict[str, Callable[[float], float]] = {
    "linear": lambda p: p,
    "cosine": lambda p: 0.5 * (1.0 - math.cos(math.pi * p)),
    "1-sqrt": lambda p: 1.0 - math.sqrt(1.0 - p),
}

# Named decays for :class:`WarmupStableDecay`'s decay phase.  Each maps
# ``progress`` ∈ ``[0, 1]`` to a *factor* in ``[0, 1]`` that multiplies
# ``(init_value - end_value)`` — 1.0 at the start of decay (value =
# init), 0.0 at the end (value = end).
_NAMED_DECAYS: dict[str, Callable[[float], float]] = {
    "linear": lambda p: 1.0 - p,
    "cosine": lambda p: 0.5 * (1.0 + math.cos(math.pi * p)),
    # Hägele et al. 2024 — concave-down, fast initial drop.
    "1-sqrt": lambda p: 1.0 - math.sqrt(p),
}


def _resolve_ramp(
    ramp: str | Callable[[float], float],
    *,
    field: str = "ramp",
) -> Callable[[float], float]:
    if callable(ramp):
        return ramp
    if ramp not in _NAMED_RAMPS:
        raise ValueError(
            f"Unknown {field}={ramp!r}; expected one of {sorted(_NAMED_RAMPS)} "
            f"or a callable."
        )
    return _NAMED_RAMPS[ramp]


def _resolve_decay(
    decay: str | Callable[[float], float],
    *,
    field: str = "decay_shape",
) -> Callable[[float], float]:
    if callable(decay):
        return decay
    if decay not in _NAMED_DECAYS:
        raise ValueError(
            f"Unknown {field}={decay!r}; expected one of "
            f"{sorted(_NAMED_DECAYS)} or a callable."
        )
    return _NAMED_DECAYS[decay]


@dataclass(frozen=True, slots=True)
class WithWarmup:
    """Multiplicative ``init_value → 1`` ramp over the first
    ``transition_steps`` steps; afterwards return
    ``schedule(step)`` unchanged.

    ``schedule`` may be a recipe (any :data:`Schedule`), a raw callable
    (won't round-trip through serialization), or a scalar ``float``
    (treated as :class:`~._curves.ConstantSchedule`).  The
    factory function :func:`with_warmup` normalises scalars at
    construction so the dataclass always stores a callable.

    ``ramp`` is one of ``"linear"``, ``"cosine"``, ``"1-sqrt"``, or a
    callable ``f(progress) -> factor in [0, 1]`` (non-serializable).
    """

    schedule: Schedule
    transition_steps: int
    ramp: Union[str, Callable[[float], float]] = "linear"
    init_value: float = 0.0

    def __call__(self, step: int) -> float:
        ramp_fn = _resolve_ramp(self.ramp)
        inner_value = self.schedule(step)
        if step >= self.transition_steps:
            return inner_value
        progress = step / self.transition_steps
        if self.init_value == 0.0:
            factor = ramp_fn(progress)
        else:
            factor = self.init_value + (1.0 - self.init_value) * ramp_fn(progress)
        return factor * inner_value


@dataclass(frozen=True, slots=True)
class WithRestarts:
    """Repeat ``schedule`` ``num_cycles`` times across
    ``[transition_begin, transition_begin + transition_steps)``.

    Each cycle has length ``transition_steps // num_cycles``; within a
    cycle, ``schedule`` is evaluated at the cycle-local integer step
    (``relative_step % cycle_length``).

    Before ``transition_begin`` returns ``schedule(0)``; after the
    final cycle, returns ``schedule(cycle_length)``.
    """

    schedule: Schedule
    transition_steps: int
    num_cycles: int
    transition_begin: int = 0

    def __call__(self, step: int) -> float:
        cycle_length = self.transition_steps // self.num_cycles
        relative = step - self.transition_begin
        if relative < 0:
            return self.schedule(0)
        if relative >= self.transition_steps:
            return self.schedule(cycle_length)
        return self.schedule(relative % cycle_length)


@dataclass(frozen=True, slots=True)
class WarmupStableDecay:
    """Three-phase schedule: warmup → constant → decay.

    Hägele et al. 2024 / MiniCPM (Hu et al. 2024) shape.  Phases over
    the ``num_warmup_steps + num_stable_steps + num_decay_steps`` total
    steps:

    1. ``[0, num_warmup_steps)`` — ramp from ``0`` to ``init_value``
       under ``warmup_ramp``.
    2. ``[num_warmup_steps, stable_end)`` — constant at ``init_value``.
    3. ``[stable_end, total)`` — decay from ``init_value`` down to
       ``end_value`` under ``decay_shape``.

    Beyond ``total`` returns ``end_value``.
    """

    init_value: float
    end_value: float
    num_warmup_steps: int
    num_stable_steps: int
    num_decay_steps: int
    warmup_ramp: Union[str, Callable[[float], float]] = "linear"
    decay_shape: Union[str, Callable[[float], float]] = "1-sqrt"

    def __call__(self, step: int) -> float:
        warmup_fn = _resolve_ramp(self.warmup_ramp, field="warmup_ramp")
        decay_fn = _resolve_decay(self.decay_shape, field="decay_shape")
        stable_end = self.num_warmup_steps + self.num_stable_steps
        total = stable_end + self.num_decay_steps
        if step < self.num_warmup_steps:
            return warmup_fn(step / self.num_warmup_steps) * self.init_value
        if step < stable_end:
            return self.init_value
        if step < total:
            p = (step - stable_end) / self.num_decay_steps
            return self.end_value + (self.init_value - self.end_value) * decay_fn(p)
        return self.end_value


# ---------------------------------------------------------------------------
# Backward-compat factory functions.
# ---------------------------------------------------------------------------


def with_warmup(
    schedule: Schedule | float,
    transition_steps: int,
    *,
    ramp: str | Callable[[float], float] = "linear",
    init_value: float = 0.0,
) -> WithWarmup:
    """Multiply ``schedule`` by an ``init_value → 1`` ramp over the
    first ``transition_steps`` steps.

    For "warmup, then decay" shapes, configure the decay with
    ``transition_begin = transition_steps``.

    A scalar ``float`` for ``schedule`` is treated as
    :class:`~._curves.ConstantSchedule`.

    ``ramp`` controls the warmup curve: ``"linear"`` (default),
    ``"cosine"``, ``"1-sqrt"``, or a callable.
    """
    if transition_steps <= 0:
        raise ValueError(
            f"with_warmup requires transition_steps > 0; got {transition_steps}."
        )
    if not 0.0 <= init_value <= 1.0:
        raise ValueError(
            f"with_warmup requires init_value in [0, 1]; got {init_value}."
        )
    # Fail-fast on bad string ramps (round-trip the resolver to validate).
    _resolve_ramp(ramp)
    inner: Schedule = (
        schedule if callable(schedule) else ConstantSchedule(float(schedule))
    )
    return WithWarmup(
        schedule=inner,
        transition_steps=int(transition_steps),
        ramp=ramp,
        init_value=float(init_value),
    )


def with_restarts(
    schedule: Schedule,
    transition_steps: int,
    num_cycles: int,
    transition_begin: int = 0,
) -> WithRestarts:
    """Repeat ``schedule`` ``num_cycles`` times across
    ``[transition_begin, transition_begin + transition_steps)``.
    """
    if num_cycles <= 0:
        raise ValueError(f"with_restarts requires num_cycles > 0; got {num_cycles}.")
    if transition_steps <= 0:
        raise ValueError(
            f"with_restarts requires transition_steps > 0; got {transition_steps}."
        )
    if transition_steps % num_cycles != 0:
        raise ValueError(
            "with_restarts requires num_cycles to evenly divide transition_steps; "
            f"got transition_steps={transition_steps}, num_cycles={num_cycles} "
            f"(remainder={transition_steps % num_cycles})."
        )
    return WithRestarts(
        schedule=schedule,
        transition_steps=int(transition_steps),
        num_cycles=int(num_cycles),
        transition_begin=int(transition_begin),
    )


def warmup_stable_decay(
    init_value: float,
    end_value: float = 0.0,
    *,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
    warmup_ramp: str | Callable[[float], float] = "linear",
    decay_shape: str | Callable[[float], float] = "1-sqrt",
) -> WarmupStableDecay:
    """Three-phase schedule: warmup → constant → decay.

    Hägele et al. 2024 / MiniCPM-shaped schedule.  See
    :class:`WarmupStableDecay` for phase details.
    """
    if num_warmup_steps <= 0:
        raise ValueError(
            f"warmup_stable_decay requires num_warmup_steps > 0; "
            f"got {num_warmup_steps}."
        )
    if num_stable_steps < 0:
        raise ValueError(
            f"warmup_stable_decay requires num_stable_steps >= 0; "
            f"got {num_stable_steps}."
        )
    if num_decay_steps <= 0:
        raise ValueError(
            f"warmup_stable_decay requires num_decay_steps > 0; got {num_decay_steps}."
        )
    # Validate named ramps/decays at construction.
    _resolve_ramp(warmup_ramp, field="warmup_ramp")
    _resolve_decay(decay_shape, field="decay_shape")
    return WarmupStableDecay(
        init_value=float(init_value),
        end_value=float(end_value),
        num_warmup_steps=int(num_warmup_steps),
        num_stable_steps=int(num_stable_steps),
        num_decay_steps=int(num_decay_steps),
        warmup_ramp=warmup_ramp,
        decay_shape=decay_shape,
    )
