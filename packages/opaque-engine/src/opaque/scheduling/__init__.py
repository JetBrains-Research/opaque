"""Step-indexed schedules + warmup / restart / WSD composition.

Each schedule is a frozen, callable recipe dataclass; the factory
functions construct the matching recipe.  See
:mod:`opaque.api.engine.scheduling` for the implementation.
"""

from opaque.api.engine.scheduling import (
    ConstantSchedule,
    CosineSchedule,
    ExponentialSchedule,
    InverseSqrtSchedule,
    LinearSchedule,
    OneMinusSqrtSchedule,
    PolynomialSchedule,
    WarmupStableDecay,
    WithRestarts,
    WithWarmup,
    constant_schedule,
    cosine_schedule,
    exponential_schedule,
    inverse_sqrt_schedule,
    linear_schedule,
    one_minus_sqrt_schedule,
    polynomial_schedule,
    warmup_stable_decay,
    with_restarts,
    with_warmup,
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
