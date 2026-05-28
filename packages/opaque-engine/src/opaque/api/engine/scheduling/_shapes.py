"""Named ramp / decay shape lookups shared by composition wrappers.

The :data:`NAMED_RAMPS` and :data:`NAMED_DECAYS` tables map a string
discriminator (``"linear"``, ``"cosine"``, ``"1-sqrt"``) to the actual
shape function.  Composition recipes (:class:`~._with_warmup.WithWarmup`,
:class:`~._warmup_stable_decay.WarmupStableDecay`) store the
discriminator string in their dataclass field and dispatch through these
helpers at ``__call__`` time — that's what keeps the recipe
serializable.

Callable overrides are accepted by the resolvers as an escape hatch for
power-user shapes; the resulting recipe is a valid Schedule at runtime
but won't round-trip through the universal serializer.
"""

from __future__ import annotations

import math
from collections.abc import Callable

__all__ = [
    "NAMED_RAMPS",
    "NAMED_DECAYS",
    "resolve_ramp",
    "resolve_decay",
]


# Named ramps (``progress in [0, 1] -> factor in [0, 1]``).
NAMED_RAMPS: dict[str, Callable[[float], float]] = {
    "linear": lambda p: p,
    "cosine": lambda p: 0.5 * (1.0 - math.cos(math.pi * p)),
    "1-sqrt": lambda p: 1.0 - math.sqrt(1.0 - p),
}

# Named decays for the decay phase of three-phase schedules.  Each maps
# ``progress`` ∈ ``[0, 1]`` to a *factor* in ``[0, 1]`` that multiplies
# ``(init_value - end_value)`` — 1.0 at the start of decay (value =
# init), 0.0 at the end (value = end).
NAMED_DECAYS: dict[str, Callable[[float], float]] = {
    "linear": lambda p: 1.0 - p,
    "cosine": lambda p: 0.5 * (1.0 + math.cos(math.pi * p)),
    # Hägele et al. 2024 — concave-down, fast initial drop.
    "1-sqrt": lambda p: 1.0 - math.sqrt(p),
}


def resolve_ramp(
    ramp: str | Callable[[float], float],
    *,
    field: str = "ramp",
) -> Callable[[float], float]:
    """Resolve ``ramp`` to a callable.

    Strings are looked up in :data:`NAMED_RAMPS`; unknown strings raise.
    Callables are returned as-is (caller takes responsibility for the
    non-serializability of recipes carrying them).
    """
    if callable(ramp):
        return ramp
    if ramp not in NAMED_RAMPS:
        raise ValueError(
            f"Unknown {field}={ramp!r}; expected one of {sorted(NAMED_RAMPS)} "
            f"or a callable."
        )
    return NAMED_RAMPS[ramp]


def resolve_decay(
    decay: str | Callable[[float], float],
    *,
    field: str = "decay_shape",
) -> Callable[[float], float]:
    """Resolve ``decay`` to a callable.  See :func:`resolve_ramp` semantics."""
    if callable(decay):
        return decay
    if decay not in NAMED_DECAYS:
        raise ValueError(
            f"Unknown {field}={decay!r}; expected one of "
            f"{sorted(NAMED_DECAYS)} or a callable."
        )
    return NAMED_DECAYS[decay]
