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

from opaque.api.engine.scheduling.types import Schedule
from opaque.api.engine.scheduling._curves import constant_schedule

__all__ = ["with_warmup", "with_restarts", "warmup_stable_decay"]


_NAMED_RAMPS: dict[str, Callable[[float], float]] = {
    "linear": lambda p: p,
    "cosine": lambda p: 0.5 * (1.0 - math.cos(math.pi * p)),
    "1-sqrt": lambda p: 1.0 - math.sqrt(1.0 - p),
}


# Named decay shapes for :func:`warmup_stable_decay`'s decay phase.
# Each maps ``progress`` ∈ ``[0, 1]`` to a *factor* in ``[0, 1]`` that
# multiplies ``(init_value - end_value)`` — 1.0 at the start of decay
# (value = init), 0.0 at the end of decay (value = end).
_NAMED_DECAYS: dict[str, Callable[[float], float]] = {
    "linear": lambda p: 1.0 - p,
    "cosine": lambda p: 0.5 * (1.0 + math.cos(math.pi * p)),
    # "1 - sqrt(progress)" — Hägele et al. 2024 ("Scaling Laws and
    # Compute-Optimal Training Beyond Fixed Training Durations").
    # Concave-down: fast initial drop, slow finish.
    "1-sqrt": lambda p: 1.0 - math.sqrt(p),
}


def with_warmup(
    schedule: Schedule | float,
    transition_steps: int,
    *,
    ramp: str | Callable[[float], float] = "linear",
    init_value: float = 0.0,
) -> Schedule:
    """Multiply ``schedule`` by an ``init_value → 1`` ramp over the
    first ``transition_steps`` steps; afterwards return
    ``schedule(step)`` unchanged.

    For the standard "warmup, then decay" shape, configure the decay
    with ``transition_begin = transition_steps``: schedules in this
    module return their ``init_value`` while
    ``step < transition_begin``, so the multiplicative ramp turns
    that plateau into a ``init_value*plateau → plateau`` warmup and
    the decay runs untouched afterwards.

    A scalar ``float`` for ``schedule`` is treated as
    :func:`~opaque.scheduling.constant_schedule`, which makes
    ``with_warmup(1e-3, transition_steps=100)`` a complete
    "warmup-then-constant" schedule.

    The ``ramp`` kwarg controls the warmup curve:

    * ``"linear"`` (default): ``progress``
    * ``"cosine"``:            ``0.5 * (1 - cos(pi * progress))``
    * ``"1-sqrt"``:            ``1 - sqrt(1 - progress)``
    * Callable ``f(progress)`` returning a factor in ``[0, 1]``.

    ``init_value`` is the starting factor for the ramp.  The default
    of ``0.0`` produces a standard ``0 → 1`` warmup so
    ``with_warmup(lr_schedule, T)`` interpolates from ``0`` up to
    ``schedule(T)``.  A positive value (e.g. ``init_value=0.1``)
    produces a "floor → 1" ramp that starts at
    ``init_value * schedule(0)`` instead of zero — useful when the
    optimizer needs a non-trivial step from the first update (e.g.
    StableAdamW), or to match ``warmup_lr_rate`` semantics from HF's
    ``cosine_warmup_with_min_lr`` schedule.  Must lie in ``[0, 1]``.

    Raises :class:`ValueError` if ``transition_steps <= 0``, ``ramp``
    is an unknown string, or ``init_value`` is outside ``[0, 1]``.
    """
    if transition_steps <= 0:
        raise ValueError(
            f"with_warmup requires transition_steps > 0; got {transition_steps}."
        )
    if not 0.0 <= init_value <= 1.0:
        raise ValueError(
            f"with_warmup requires init_value in [0, 1]; got {init_value}."
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
        # Fast path: ``factor = ramp(p)`` directly, matching the original
        # zero-floor behaviour.
        def wrapped(step: int) -> float:
            if step < transition_steps:
                return ramp_fn(step / transition_steps) * inner(step)
            return inner(step)
    else:
        # ``factor = init_value + (1 - init_value) * ramp(p)`` interpolates
        # from ``init_value`` at progress=0 up to ``1.0`` at progress=1.
        delta = 1.0 - init_value

        def wrapped(step: int) -> float:
            if step < transition_steps:
                p = step / transition_steps
                factor = init_value + delta * ramp_fn(p)
                return factor * inner(step)
            return inner(step)

    return wrapped


def warmup_stable_decay(
    init_value: float,
    end_value: float = 0.0,
    *,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
    warmup_ramp: str | Callable[[float], float] = "linear",
    decay_shape: str | Callable[[float], float] = "1-sqrt",
) -> Schedule:
    """Three-phase schedule: warmup → constant → decay.

    Standard LLM pre-training shape from Hägele et al. 2024 ("Scaling
    Laws and Compute-Optimal Training Beyond Fixed Training
    Durations") and the MiniCPM paper (Hu et al. 2024).  The stable
    middle plateau enables "decay-only fine-tuning": resume from any
    checkpoint in the stable region and run only the decay tail
    without re-training the warmup.

    Phases over the
    ``num_warmup_steps + num_stable_steps + num_decay_steps`` total
    steps:

    1. ``[0, num_warmup_steps)`` — ramp from ``0`` to ``init_value``
       under ``warmup_ramp`` (linear by default).
    2. ``[num_warmup_steps, num_warmup_steps + num_stable_steps)`` —
       constant at ``init_value``.
    3. ``[num_warmup_steps + num_stable_steps, total)`` — decay from
       ``init_value`` down to ``end_value`` under ``decay_shape``.

    Beyond ``total`` returns ``end_value``.

    Args:
        init_value: Peak learning rate (held during the stable phase).
        end_value: Decay target.  Defaults to ``0.0``; pass a positive
            value (e.g. ``0.1 * init_value``) for a non-zero "min_lr"
            floor.
        num_warmup_steps: Length of the warmup ramp.  Must be ``> 0``.
        num_stable_steps: Length of the constant plateau.  Must be
            ``>= 0`` — pass ``0`` to skip the stable phase (the result
            is a pure warmup-then-decay schedule).
        num_decay_steps: Length of the decay tail.  Must be ``> 0``.
        warmup_ramp: Curve for the warmup phase.  Accepts the same
            values as :func:`with_warmup` — ``"linear"``, ``"cosine"``,
            ``"1-sqrt"``, or a callable.
        decay_shape: Curve for the decay phase.  Defaults to
            ``"1-sqrt"`` (the WSD paper's recommendation): factor
            ``1 - sqrt(progress)``, concave-down — fast initial drop,
            slow finish.  Also accepts ``"linear"``, ``"cosine"``
            (half-cosine from init to end), or a callable
            ``f(progress) -> factor in [0, 1]`` where the factor
            multiplies ``(init_value - end_value)`` (1.0 at start of
            decay, 0.0 at end).

    Raises:
        :class:`ValueError`: if any phase length is invalid, or if
            ``warmup_ramp`` / ``decay_shape`` is an unknown string.
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
            f"warmup_stable_decay requires num_decay_steps > 0; "
            f"got {num_decay_steps}."
        )

    if isinstance(warmup_ramp, str):
        if warmup_ramp not in _NAMED_RAMPS:
            raise ValueError(
                f"Unknown warmup_ramp={warmup_ramp!r}; expected one of "
                f"{sorted(_NAMED_RAMPS)} or a callable."
            )
        warmup_fn = _NAMED_RAMPS[warmup_ramp]
    else:
        warmup_fn = warmup_ramp

    if isinstance(decay_shape, str):
        if decay_shape not in _NAMED_DECAYS:
            raise ValueError(
                f"Unknown decay_shape={decay_shape!r}; expected one of "
                f"{sorted(_NAMED_DECAYS)} or a callable."
            )
        decay_fn = _NAMED_DECAYS[decay_shape]
    else:
        decay_fn = decay_shape

    delta = init_value - end_value
    stable_end = num_warmup_steps + num_stable_steps
    total = stable_end + num_decay_steps

    def wrapped(step: int) -> float:
        if step < num_warmup_steps:
            return warmup_fn(step / num_warmup_steps) * init_value
        if step < stable_end:
            return init_value
        if step < total:
            p = (step - stable_end) / num_decay_steps
            return end_value + delta * decay_fn(p)
        return end_value

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
