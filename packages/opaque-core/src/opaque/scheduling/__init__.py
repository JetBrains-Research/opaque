"""Step-indexed scalar schedules and warmup composition.

Each public function returns a plain ``Callable[[int], float]``
suitable as the ``lr`` argument to a functional optimizer::

    from opaque.scheduling import cosine_schedule, with_warmup

    decay = cosine_schedule(
        init_value=1e-3, end_value=0.0,
        transition_steps=900, transition_begin=100,
    )
    schedule = with_warmup(decay, transition_steps=100)
    # `schedule(step)` returns the scheduled value at any integer step.

The :data:`Schedule` type alias is reachable via
:mod:`opaque.scheduling.types`.
"""

from opaque.scheduling._compose import with_restarts, with_warmup
from opaque.scheduling._curves import (
    constant_schedule,
    cosine_schedule,
    exponential_schedule,
    inverse_sqrt_schedule,
    linear_schedule,
    one_minus_sqrt_schedule,
    polynomial_schedule,
)

__all__ = [
    "constant_schedule",
    "linear_schedule",
    "polynomial_schedule",
    "exponential_schedule",
    "cosine_schedule",
    "inverse_sqrt_schedule",
    "one_minus_sqrt_schedule",
    "with_warmup",
    "with_restarts",
]
