"""Materialized learning-rate schedule identity for strategy caches."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opaque.api.engine.scheduling.types import Schedule


def materialize_schedule(
    lr_schedule: Schedule | None, n_steps: int
) -> tuple[float, ...] | None:
    """Return the schedule values that affect an ``n_steps`` strategy query."""
    if lr_schedule is None:
        return None
    return tuple(float(lr_schedule(step)) for step in range(n_steps))


def strategy_schedule_fingerprint(
    strategy: Any, n_steps: int
) -> tuple[type[Any], tuple[float, ...] | None]:
    """Return the privacy-material schedule identity for a strategy query."""
    return (
        type(strategy),
        materialize_schedule(getattr(strategy, "lr_schedule", None), n_steps),
    )
