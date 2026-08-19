"""Schedule-free averaging wrapper for engine optimizer factories."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opaque.api.engine import ops
from opaque.api.engine.optimizers.types import ScheduleFreeState
from opaque.pytree import tree_map

if TYPE_CHECKING:
    from collections.abc import Callable


def schedule_free(
    params: Any,
    base: Callable[..., tuple[Callable[..., tuple[Any, Any]], Any]],
    *,
    beta: float = 0.9,
    warmup_steps: int = 0,
    **base_kwargs: Any,
) -> tuple[Callable[..., tuple[Any, ScheduleFreeState]], ScheduleFreeState]:
    """Wrap an optimizer factory while retaining raw and published weights.

    ``base`` receives ``params`` followed by ``base_kwargs`` and must follow
    the engine optimizer factory protocol.  The returned state exposes ``x``
    for evaluation/checkpointing and the step returns the signed delta from
    the training weights ``y`` to the next interpolation.
    """
    if not 0 <= beta <= 1:
        raise ValueError("beta must satisfy 0 <= beta <= 1")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    inner_step, inner_state = base(params, **base_kwargs)
    state = ScheduleFreeState(
        tree_map(ops.clone, params), tree_map(ops.clone, params), inner_state, 0, beta
    )

    def step(
        grads: Any, state: ScheduleFreeState, *, params: Any
    ) -> tuple[Any, ScheduleFreeState]:
        inner_update, inner = inner_step(grads, state.inner, params=state.z)
        z = tree_map(ops.add, state.z, inner_update)
        t = state.step + 1
        if state.step < warmup_steps:
            x = tree_map(ops.clone, z)
        else:
            weight = 1.0 / (t - warmup_steps)
            x = tree_map(
                lambda old, value: ops.add(
                    ops.multiply(old, 1 - weight), ops.multiply(value, weight)
                ),
                state.x,
                z,
            )
        y = tree_map(
            lambda raw, average: ops.add(
                ops.multiply(raw, 1 - state.beta), ops.multiply(average, state.beta)
            ),
            z,
            x,
        )
        return tree_map(ops.subtract, y, params), ScheduleFreeState(
            z, x, inner, t, state.beta
        )

    return step, state


__all__ = ["schedule_free"]
