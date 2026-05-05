"""Composition primitives for step-indexed schedules.

These take one or more :data:`~opaque.scheduling.Schedule` objects and
return a new :data:`~opaque.scheduling.Schedule`.  They are the
"glue" layer that lets the curves in :mod:`opaque.scheduling._curves`
be combined into the standard "warmup, then decay", "cosine with
restarts", etc. shapes.
"""

from __future__ import annotations

import math
from typing import Callable

from opaque.scheduling.types import Schedule
from opaque.scheduling._curves import constant_schedule

__all__ = ["with_warmup", "with_restarts"]


_NAMED_RAMPS: dict[str, Callable[[float], float]] = {
    "linear": lambda p: p,
    "cosine": lambda p: 0.5 * (1.0 - math.cos(math.pi * p)),
    "1-sqrt": lambda p: 1.0 - math.sqrt(1.0 - p),
}


def with_warmup(
    schedule: Schedule | float,
    transition_steps: int,
    *,
    ramp: str | Callable[[float], float] = "linear",
    init_value: float = 0.0,
) -> Schedule:
    """Multiply ``schedule`` by an ``init_value`` → 1 ramp over the
    first ``transition_steps`` steps; afterwards return
    ``schedule(step)`` unchanged.

    For the standard "warmup, then decay" shape, configure the decay
    with ``transition_begin = transition_steps``: schedules in this
    module return their ``init_value`` while
    ``step < transition_begin``, so the multiplicative ramp turns
    that plateau into an ``init_value * inner_init`` →
    ``inner_init`` warmup and the decay runs untouched afterwards.

    A scalar ``float`` for ``schedule`` is treated as
    :func:`~opaque.scheduling.constant_schedule`, which makes
    ``with_warmup(1e-3, transition_steps=100)`` a complete
    "warmup-then-constant" schedule.

    The ``ramp`` kwarg controls the warmup curve shape:

    * ``"linear"`` (default): ``progress``
    * ``"cosine"``:            ``0.5 * (1 - cos(pi * progress))``
    * ``"1-sqrt"``:            ``1 - sqrt(1 - progress)``
    * Callable ``f(progress)`` returning a factor in ``[0, 1]``.

    ``init_value`` sets the factor at step 0; the ramp ends at 1.0
    over ``transition_steps`` and the inner schedule runs unchanged
    afterwards.  The factor at intermediate steps is
    ``init_value + (1 - init_value) * ramp(progress)``.

    Raises :class:`ValueError` if ``transition_steps <= 0`` or ``ramp``
    is an unknown string.
    """
    if transition_steps <= 0:
        raise ValueError(
            f"with_warmup requires transition_steps > 0; got {transition_steps}."
        )
    if isinstance(ramp, str):
        if ramp not in _NAMED_RAMPS:
            raise ValueError(
                f"Unknown ramp={ramp!r}; expected one of {sorted(_NAMED_RAMPS)} "
                f"or a callable."
            )
        ramp_fn = _NAMED_RAMPS[ramp]
    else:
        ramp_fn = ramp

    inner: Schedule = schedule if callable(schedule) else constant_schedule(schedule)

    if init_value == 0.0:
        def wrapped(step: int) -> float:
            if step < transition_steps:
                return ramp_fn(step / transition_steps) * inner(step)
            return inner(step)
    else:
        span = 1.0 - init_value

        def wrapped(step: int) -> float:
            if step < transition_steps:
                factor = init_value + span * ramp_fn(step / transition_steps)
                return factor * inner(step)
            return inner(step)

    return wrapped


def with_restarts(
    schedule: Schedule,
    transition_steps: int,
    num_cycles: int,
    transition_begin: int = 0,
) -> Schedule:
    """Repeat ``schedule`` ``num_cycles`` times across
    ``[transition_begin, transition_begin + transition_steps)``.

    Each cycle has length ``transition_steps // num_cycles``; within a
    cycle, ``schedule`` is evaluated at the cycle-local integer step
    (``relative_step % cycle_length``).  Configure ``schedule`` to
    produce its full curve over a single cycle of that length.

    Before ``transition_begin`` returns ``schedule(0)``; after the
    final cycle, returns ``schedule(cycle_length)``.

    Raises :class:`ValueError` if ``num_cycles <= 0``,
    ``transition_steps <= 0``, or ``num_cycles`` does not evenly divide
    ``transition_steps`` (the cycle length must be an integer so the
    inner schedule receives integer steps).
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

    cycle_length = transition_steps // num_cycles

    def wrapped(step: int) -> float:
        relative = step - transition_begin
        if relative < 0:
            return schedule(0)
        if relative >= transition_steps:
            return schedule(cycle_length)
        return schedule(relative % cycle_length)

    return wrapped
