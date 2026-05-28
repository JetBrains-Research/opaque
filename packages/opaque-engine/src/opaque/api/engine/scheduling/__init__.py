"""Step-indexed scalar schedules — factory functions.

Curves and composition wrappers are frozen, callable recipe dataclasses;
each factory returns the matching recipe instance.  Call-site usage is
unchanged from the previous closure-based API::

    from opaque.api.engine.scheduling import cosine_schedule, with_warmup

    decay = cosine_schedule(
        init_value=1e-3, end_value=0.0,
        transition_steps=900, transition_begin=100,
    )
    schedule = with_warmup(decay, transition_steps=100)
    schedule(step)   # → float, scheduled value at integer step

The recipe classes themselves live in
:mod:`opaque.api.engine.scheduling.types` for ``isinstance`` checks and
type annotations.
"""

from opaque.api.engine.scheduling._compose import (
    warmup_stable_decay,
    with_restarts,
    with_warmup,
)
from opaque.api.engine.scheduling._curves import (
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
    "warmup_stable_decay",
]
