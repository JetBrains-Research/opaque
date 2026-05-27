"""Step-indexed scalar curves as frozen, callable recipe dataclasses.

Each curve is a ``@dataclass(frozen=True)`` whose fields are the factory
arguments and whose ``__call__(step)`` evaluates the curve.  Frozen
dataclasses give us value-based equality + hashing for free, which
keeps downstream LRU caches stable across processes and lets the
universal :mod:`opaque.serialization` registry round-trip the recipe
without any per-class handler — every field is a primitive.

The module-level factory functions (``constant_schedule`` etc.) are kept
for backward compatibility and now just construct + return the matching
dataclass instance.  Existing call-sites that treat the result as a
``Callable[[int], float]`` keep working unchanged because the
dataclasses implement ``__call__``.

For "warmup, then decay" shapes, configure the decay with
``transition_begin = num_warmup_steps`` and wrap with
:class:`~opaque.api.engine.scheduling.WithWarmup`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from opaque.api.engine.scheduling.types import Schedule  # noqa: F401  (re-exported)

__all__ = [
    "ConstantSchedule",
    "LinearSchedule",
    "PolynomialSchedule",
    "ExponentialSchedule",
    "CosineSchedule",
    "InverseSqrtSchedule",
    "OneMinusSqrtSchedule",
    "constant_schedule",
    "linear_schedule",
    "polynomial_schedule",
    "exponential_schedule",
    "cosine_schedule",
    "inverse_sqrt_schedule",
    "one_minus_sqrt_schedule",
]


@dataclass(frozen=True, slots=True)
class ConstantSchedule:
    """``schedule(step) == value`` for every step."""

    value: float

    def __call__(self, _step: int) -> float:
        return self.value


@dataclass(frozen=True, slots=True)
class PolynomialSchedule:
    """Polynomial transition from ``init_value`` to ``end_value``.

    ``end + (init - end) * (1 - count/T)^power`` where
    ``count = clamp(step - transition_begin, 0, T)``.

    Steps before ``transition_begin`` hold at ``init_value``; steps after
    ``transition_begin + transition_steps`` hold at ``end_value``.
    """

    init_value: float
    end_value: float
    power: float
    transition_steps: int
    transition_begin: int = 0

    def __call__(self, step: int) -> float:
        span = max(1, self.transition_steps)
        count = max(0, min(span, step - self.transition_begin))
        frac = 1.0 - count / span
        return (self.init_value - self.end_value) * (frac**self.power) + self.end_value


@dataclass(frozen=True, slots=True)
class LinearSchedule:
    """Linear interpolation from ``init_value`` to ``end_value`` over
    ``transition_steps``.  Equivalent to
    :class:`PolynomialSchedule` with ``power=1.0``.
    """

    init_value: float
    end_value: float
    transition_steps: int
    transition_begin: int = 0

    def __call__(self, step: int) -> float:
        # Delegate to PolynomialSchedule's arithmetic to keep the
        # single-source-of-truth invariant.  No allocation in the hot
        # path — the polynomial body is inlined here.
        span = max(1, self.transition_steps)
        count = max(0, min(span, step - self.transition_begin))
        frac = 1.0 - count / span
        return (self.init_value - self.end_value) * frac + self.end_value


@dataclass(frozen=True, slots=True)
class ExponentialSchedule:
    """``init * decay_rate^((step - transition_begin) / transition_steps)``.

    Direction is set by ``decay_rate``: ``< 1`` decay, ``> 1`` growth,
    ``== 1`` constant.  Steps before ``transition_begin`` hold at
    ``init_value``.  When ``staircase`` is true the exponent is floored
    to an integer.  ``end_value`` optionally clamps the result (lower
    bound for decay, upper bound for growth).
    """

    init_value: float
    decay_rate: float
    transition_begin: int = 0
    transition_steps: int = 1
    staircase: bool = False
    end_value: float | None = None

    def __call__(self, step: int) -> float:
        if self.transition_steps <= 0 or self.decay_rate == 0:
            return float(self.init_value)
        decreased = step - self.transition_begin
        if decreased <= 0:
            return float(self.init_value)
        p = decreased / self.transition_steps
        if self.staircase:
            p = math.floor(p)
        decayed = self.init_value * (self.decay_rate**p)
        if self.end_value is not None:
            clip = max if self.decay_rate < 1.0 else min
            return clip(decayed, self.end_value)
        return decayed


@dataclass(frozen=True, slots=True)
class CosineSchedule:
    """Cosine annealing from ``init_value`` to ``end_value``.

    ``progress = (step - transition_begin) / transition_steps`` grows
    past 1 and the cosine continues to oscillate — the ``max(0, cos)``
    clip keeps the schedule non-negative.  With ``num_cycles=0.5``
    (default) this is a single half-cosine bottoming out at
    ``end_value`` when ``progress == 1``; larger values produce
    additional oscillations.
    """

    init_value: float
    end_value: float
    transition_steps: int
    transition_begin: int = 0
    num_cycles: float = 0.5

    def __call__(self, step: int) -> float:
        span = max(1, self.transition_steps)
        progress = max(0, step - self.transition_begin) / span
        cos = 0.5 * (1.0 + math.cos(math.pi * self.num_cycles * 2.0 * progress))
        return self.end_value + (self.init_value - self.end_value) * max(0.0, cos)


@dataclass(frozen=True, slots=True)
class InverseSqrtSchedule:
    """``init_value * sqrt(T / (s + T))`` where
    ``T = transition_steps`` and ``s = max(0, step - transition_begin)``.

    At ``s=0`` returns ``init_value``; at ``s=T`` returns
    ``init_value / sqrt(2)``.
    """

    init_value: float
    transition_steps: int
    transition_begin: int = 0

    def __call__(self, step: int) -> float:
        span = max(1, self.transition_steps)
        s = max(0, step - self.transition_begin)
        return self.init_value * math.sqrt(span / (s + span))


@dataclass(frozen=True, slots=True)
class OneMinusSqrtSchedule:
    """Decay following ``factor = 1 - sqrt(progress)`` from
    ``init_value`` at ``transition_begin`` to ``end_value`` at
    ``transition_begin + transition_steps``.

    Concave decreasing — the value drops faster early than late.
    """

    init_value: float
    end_value: float
    transition_steps: int
    transition_begin: int = 0

    def __call__(self, step: int) -> float:
        span = max(1, self.transition_steps)
        progress = min(1.0, max(0, step - self.transition_begin) / span)
        factor = 1.0 - math.sqrt(progress)
        return self.end_value + (self.init_value - self.end_value) * factor


# ---------------------------------------------------------------------------
# Backward-compat factory functions.
# ---------------------------------------------------------------------------
#
# These predate the recipe refactor.  They now just construct the matching
# recipe dataclass and return it.  Callers that treat the result as a
# ``Callable[[int], float]`` continue to work unchanged because each
# recipe implements ``__call__``.


def constant_schedule(value: float) -> ConstantSchedule:
    """Return a schedule that yields ``value`` at every step."""
    return ConstantSchedule(value=float(value))


def linear_schedule(
    init_value: float,
    end_value: float,
    transition_steps: int,
    transition_begin: int = 0,
) -> LinearSchedule:
    """Linear interpolation from ``init_value`` to ``end_value`` over
    ``transition_steps``, starting at ``transition_begin``.

    Steps before ``transition_begin`` hold at ``init_value``; steps
    after ``transition_begin + transition_steps`` hold at
    ``end_value``.
    """
    return LinearSchedule(
        init_value=float(init_value),
        end_value=float(end_value),
        transition_steps=int(transition_steps),
        transition_begin=int(transition_begin),
    )


def polynomial_schedule(
    init_value: float,
    end_value: float,
    power: float,
    transition_steps: int,
    transition_begin: int = 0,
) -> PolynomialSchedule:
    """Polynomial transition from ``init_value`` to ``end_value`` over
    ``transition_steps``: ``end + (init - end) * (1 - count/T)^power``.

    Steps before ``transition_begin`` hold at ``init_value``; steps
    after ``transition_begin + transition_steps`` hold at
    ``end_value``.
    """
    return PolynomialSchedule(
        init_value=float(init_value),
        end_value=float(end_value),
        power=float(power),
        transition_steps=int(transition_steps),
        transition_begin=int(transition_begin),
    )


def exponential_schedule(
    init_value: float,
    decay_rate: float,
    transition_begin: int = 0,
    transition_steps: int = 1,
    staircase: bool = False,
    end_value: float | None = None,
) -> ExponentialSchedule:
    """Exponential schedule: ``init * decay_rate^((step - transition_begin) / transition_steps)``."""
    return ExponentialSchedule(
        init_value=float(init_value),
        decay_rate=float(decay_rate),
        transition_begin=int(transition_begin),
        transition_steps=int(transition_steps),
        staircase=bool(staircase),
        end_value=None if end_value is None else float(end_value),
    )


def cosine_schedule(
    init_value: float,
    end_value: float,
    transition_steps: int,
    transition_begin: int = 0,
    num_cycles: float = 0.5,
) -> CosineSchedule:
    """Cosine annealing from ``init_value`` to ``end_value`` over ``transition_steps``."""
    return CosineSchedule(
        init_value=float(init_value),
        end_value=float(end_value),
        transition_steps=int(transition_steps),
        transition_begin=int(transition_begin),
        num_cycles=float(num_cycles),
    )


def inverse_sqrt_schedule(
    init_value: float,
    transition_steps: int,
    transition_begin: int = 0,
) -> InverseSqrtSchedule:
    """Inverse-square-root decay: ``init_value * sqrt(T / (s + T))``."""
    return InverseSqrtSchedule(
        init_value=float(init_value),
        transition_steps=int(transition_steps),
        transition_begin=int(transition_begin),
    )


def one_minus_sqrt_schedule(
    init_value: float,
    end_value: float,
    transition_steps: int,
    transition_begin: int = 0,
) -> OneMinusSqrtSchedule:
    """Decay following ``factor = 1 - sqrt(progress)``."""
    return OneMinusSqrtSchedule(
        init_value=float(init_value),
        end_value=float(end_value),
        transition_steps=int(transition_steps),
        transition_begin=int(transition_begin),
    )
