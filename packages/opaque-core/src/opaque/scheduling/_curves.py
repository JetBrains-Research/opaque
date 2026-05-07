"""Pure step-indexed scalar curves.

Each function returns a :data:`~opaque.scheduling.Schedule` —
``Callable[[int], float]``.

For "warmup, then decay" shapes, configure the decay with
``transition_begin = num_warmup_steps`` and wrap with
:func:`opaque.scheduling.with_warmup`.
"""

from __future__ import annotations

import math

from opaque.scheduling.types import Schedule

__all__ = [
    "constant_schedule",
    "linear_schedule",
    "polynomial_schedule",
    "exponential_schedule",
    "cosine_schedule",
    "inverse_sqrt_schedule",
    "one_minus_sqrt_schedule",
]


def constant_schedule(value: float) -> Schedule:
    """Return a schedule that yields ``value`` at every step."""
    return lambda _step: value


def linear_schedule(
    init_value: float,
    end_value: float,
    transition_steps: int,
    transition_begin: int = 0,
) -> Schedule:
    """Linear interpolation from ``init_value`` to ``end_value`` over
    ``transition_steps``, starting at ``transition_begin``.

    Steps before ``transition_begin`` hold at ``init_value``; steps
    after ``transition_begin + transition_steps`` hold at
    ``end_value``.
    """
    return polynomial_schedule(
        init_value,
        end_value,
        1.0,
        transition_steps,
        transition_begin,
    )


def polynomial_schedule(
    init_value: float,
    end_value: float,
    power: float,
    transition_steps: int,
    transition_begin: int = 0,
) -> Schedule:
    """Polynomial transition from ``init_value`` to ``end_value`` over
    ``transition_steps``: ``end + (init - end) * (1 - count/T)^power``.

    Steps before ``transition_begin`` hold at ``init_value``; steps
    after ``transition_begin + transition_steps`` hold at
    ``end_value``.
    """
    span = max(1, transition_steps)
    delta = init_value - end_value

    def schedule(step: int) -> float:
        count = max(0, min(span, step - transition_begin))
        frac = 1.0 - count / span
        return delta * (frac**power) + end_value

    return schedule


def exponential_schedule(
    init_value: float,
    decay_rate: float,
    transition_begin: int = 0,
    transition_steps: int = 1,
    staircase: bool = False,
    end_value: float | None = None,
) -> Schedule:
    """Exponential schedule: ``init * decay_rate^((step - transition_begin) / transition_steps)``.

    The shape is exponential; the direction is set by ``decay_rate``:
    ``decay_rate < 1`` produces decay, ``decay_rate > 1`` produces
    growth, ``decay_rate == 1`` produces a constant.  Steps before
    ``transition_begin`` hold at ``init_value``.  When ``staircase`` is
    ``True``, the exponent is floored to integer so the schedule moves
    in discrete jumps.  ``end_value`` clamps the result (lower bound
    for ``decay_rate < 1``, upper bound for ``decay_rate > 1``).
    """
    if transition_steps <= 0 or decay_rate == 0:
        return lambda _step: init_value

    def schedule(step: int) -> float:
        decreased = step - transition_begin
        if decreased <= 0:
            return float(init_value)
        p = decreased / transition_steps
        if staircase:
            p = math.floor(p)
        decayed = init_value * (decay_rate**p)
        if end_value is not None:
            clip = max if decay_rate < 1.0 else min
            return clip(decayed, end_value)
        return decayed

    return schedule


def cosine_schedule(
    init_value: float,
    end_value: float,
    transition_steps: int,
    transition_begin: int = 0,
    num_cycles: float = 0.5,
) -> Schedule:
    """Cosine annealing from ``init_value`` to ``end_value`` over ``transition_steps``.

    Steps before ``transition_begin`` hold at ``init_value``.  After
    ``transition_begin``, ``progress = (step - transition_begin) / transition_steps``
    grows past 1 and the cosine continues to oscillate — the
    ``max(0, cos)`` clip guarantees the schedule never goes negative.
    With the default ``num_cycles=0.5`` this is a single half-cosine
    that bottoms out at ``end_value`` exactly when ``progress = 1``;
    larger values produce additional oscillations.
    """
    span = max(1, transition_steps)
    delta = init_value - end_value

    def schedule(step: int) -> float:
        progress = max(0, step - transition_begin) / span
        cos = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        return end_value + delta * max(0.0, cos)

    return schedule


def inverse_sqrt_schedule(
    init_value: float,
    transition_steps: int,
    transition_begin: int = 0,
) -> Schedule:
    """Inverse-square-root decay: ``init_value * sqrt(T / (s + T))``.

    Here ``T = transition_steps`` is the timescale and
    ``s = max(0, step - transition_begin)``.  At ``s=0`` returns
    ``init_value``; at ``s=T`` returns ``init_value / sqrt(2)``.
    """
    span = max(1, transition_steps)

    def schedule(step: int) -> float:
        s = max(0, step - transition_begin)
        return init_value * math.sqrt(span / (s + span))

    return schedule


def one_minus_sqrt_schedule(
    init_value: float,
    end_value: float,
    transition_steps: int,
    transition_begin: int = 0,
) -> Schedule:
    """Decay following ``factor = 1 - sqrt(progress)`` from
    ``init_value`` at ``transition_begin`` to ``end_value`` at
    ``transition_begin + transition_steps``.

    Concave decreasing — the value drops faster early than late.
    """
    span = max(1, transition_steps)
    delta = init_value - end_value

    def schedule(step: int) -> float:
        progress = max(0, step - transition_begin) / span
        progress = min(1.0, progress)
        factor = 1.0 - math.sqrt(progress)
        return end_value + delta * factor

    return schedule
