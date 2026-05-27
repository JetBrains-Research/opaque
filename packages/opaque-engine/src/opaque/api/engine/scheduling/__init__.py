"""Step-indexed scalar schedules — recipe dataclasses + factory functions.

Each curve and composition wrapper is a ``@dataclass(frozen=True)``
implementing ``__call__(step) -> float`` so the recipe is both the
configuration and the schedule itself.  Backward-compatible factory
functions (``cosine_schedule(...)``, ``with_warmup(...)``) construct
and return the matching recipe — call-site usage is unchanged::

    from opaque.scheduling import cosine_schedule, with_warmup

    decay = cosine_schedule(
        init_value=1e-3, end_value=0.0,
        transition_steps=900, transition_begin=100,
    )
    schedule = with_warmup(decay, transition_steps=100)
    schedule(step)   # → float, scheduled value at integer step

Recipes are frozen dataclasses over primitive fields (plus nested
recipes for composition), so they round-trip cleanly through
:mod:`opaque.serialization` without any custom codec — useful for
checkpointing strategies that take a schedule (BandMF / BLT) and
optimizer state that holds a schedule reference.

The :data:`Schedule` type alias is kept as ``Callable[[int], float]``;
every recipe satisfies it via ``__call__`` and user-supplied raw
callables (lambdas etc.) are accepted by every consumer at the cost of
not being round-trippable.
"""

from opaque.api.engine.scheduling._compose import (
    WarmupStableDecay,
    WithRestarts,
    WithWarmup,
    warmup_stable_decay,
    with_restarts,
    with_warmup,
)
from opaque.api.engine.scheduling._curves import (
    ConstantSchedule,
    CosineSchedule,
    ExponentialSchedule,
    InverseSqrtSchedule,
    LinearSchedule,
    OneMinusSqrtSchedule,
    PolynomialSchedule,
    constant_schedule,
    cosine_schedule,
    exponential_schedule,
    inverse_sqrt_schedule,
    linear_schedule,
    one_minus_sqrt_schedule,
    polynomial_schedule,
)

__all__ = [
    # Recipe classes
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
    # Factory functions (backward-compat)
    "constant_schedule",
    "linear_schedule",
    "polynomial_schedule",
    "exponential_schedule",
    "cosine_schedule",
    "inverse_sqrt_schedule",
    "one_minus_sqrt_schedule",
    "with_warmup",
    "with_restarts",
    "warmup_stable_decay",
]
