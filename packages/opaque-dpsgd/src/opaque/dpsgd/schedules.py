"""Learning-rate schedules for functional (torchopt) optimizers.

Each public function returns a plain ``Callable[[int], float]`` that
can be passed directly as the ``lr`` argument to ``torchopt.adamw`` /
``adam`` / ``sgd`` / ``opaque.dpsgd.adamw_bc``::

    from opaque.dpsgd.schedules import cosine_schedule, with_warmup
    import torchopt

    # Cosine that plateaus during [0, 100), then decays over [100, 1000).
    decay = cosine_schedule(
        init_value=1e-3, end_value=0.0,
        transition_steps=900, transition_begin=100,
    )
    schedule = with_warmup(decay, transition_steps=100)
    opt = torchopt.adamw(lr=schedule)

``linear_schedule``, ``polynomial_schedule`` and ``exponential_decay``
are re-exported from :mod:`torchopt.schedule` so a complete schedule
can be assembled from a single import path.
"""

from __future__ import annotations

import math
from typing import Callable

from torchopt.schedule import (
    exponential_decay,
    linear_schedule,
    polynomial_schedule,
)

__all__ = [
    # Re-exports from torchopt.schedule
    "linear_schedule",
    "polynomial_schedule",
    "exponential_decay",
    # Pure curves
    "constant_schedule",
    "cosine_schedule",
    "inverse_sqrt_schedule",
    "one_minus_sqrt_schedule",
    # Composition primitives
    "with_warmup",
    "with_restarts",
]


Schedule = Callable[[int], float]


def constant_schedule(value: float) -> Schedule:
    """Return a schedule that yields ``value`` at every step."""
    return lambda _step: value


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
    grows past 1 and the cosine continues to oscillate — the ``max(0, cos)``
    clip guarantees the schedule never goes negative.  With the default
    ``num_cycles=0.5`` this is a single half-cosine that bottoms out at
    ``end_value`` exactly when ``progress = 1``; larger values produce
    additional oscillations.
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

    Concave decreasing — the LR drops faster early than late.
    """
    span = max(1, transition_steps)
    delta = init_value - end_value

    def schedule(step: int) -> float:
        progress = max(0, step - transition_begin) / span
        progress = min(1.0, progress)
        factor = 1.0 - math.sqrt(progress)
        return end_value + delta * factor

    return schedule


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
) -> Schedule:
    """Multiply ``schedule`` by a 0 → 1 ramp over the first
    ``transition_steps`` steps; afterwards return ``schedule(step)``
    unchanged.

    For the standard "warmup, then decay" shape, configure the decay
    with ``transition_begin = transition_steps``: schedules in this
    module and in :mod:`torchopt.schedule` return their ``init_value``
    while ``step < transition_begin``, so the multiplicative ramp turns
    that plateau into a 0 → ``init_value`` warmup and the decay runs
    untouched afterwards.

    A scalar ``float`` for ``schedule`` is treated as
    :func:`constant_schedule`, which makes ``with_warmup(1e-3,
    transition_steps=100)`` a complete "warmup-then-constant" schedule.

    The ``ramp`` kwarg controls the warmup curve:

    * ``"linear"`` (default): ``progress``
    * ``"cosine"``:            ``0.5 * (1 - cos(pi * progress))``
    * ``"1-sqrt"``:            ``1 - sqrt(1 - progress)``
    * Callable ``f(progress)`` returning a factor in ``[0, 1]``.

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

    def wrapped(step: int) -> float:
        if step < transition_steps:
            return ramp_fn(step / transition_steps) * inner(step)
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

    Each cycle has length ``transition_steps / num_cycles``; within a
    cycle, ``schedule`` is evaluated at the cycle-local step
    (``relative_step % cycle_length``).  Configure ``schedule`` to
    produce its full curve over a single cycle of that length.

    Before ``transition_begin`` returns ``schedule(0)``; after the
    final cycle, returns ``schedule(cycle_length)``.

    Raises :class:`ValueError` if ``num_cycles <= 0`` or
    ``transition_steps <= 0``.
    """
    if num_cycles <= 0:
        raise ValueError(
            f"with_restarts requires num_cycles > 0; got {num_cycles}."
        )
    if transition_steps <= 0:
        raise ValueError(
            f"with_restarts requires transition_steps > 0; got {transition_steps}."
        )

    cycle_length = transition_steps / num_cycles

    def wrapped(step: int) -> float:
        relative = step - transition_begin
        if relative < 0:
            return schedule(0)
        if relative >= transition_steps:
            return schedule(cycle_length)
        return schedule(relative % cycle_length)

    return wrapped
