"""Backend-neutral Lion optimizer."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from opaque.api.engine import ops
from opaque.api.engine.optimizers._chain import make_optimizer_chain
from opaque.api.engine.optimizers.types import LionState
from opaque.pytree import tree_map

_LR = float | Callable[[int], float]


def lion(
    params: Any,
    lr: _LR = 1e-4,
    betas: tuple[float, float] = (0.9, 0.99),
    weight_decay: float = 0.0,
    *,
    decoupled_weight_decay: bool = True,
) -> tuple[Callable[..., tuple[Any, LionState]], LionState]:
    if len(betas) != 2 or not all(0 <= beta < 1 for beta in betas):
        raise ValueError(f"betas must contain two values in [0, 1), got {betas}")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    b1, b2 = betas
    initial = LionState(tree_map(ops.zeros_like, params), 0)

    def moment(
        grads: Any, state: LionState, _params: Any, _stddev: Any, _squared: Any
    ) -> tuple[Any, LionState]:
        def sign(m: Any, g: Any) -> Any:
            value = ops.add(ops.multiply(m, b1), ops.multiply(g, 1 - b1))
            # (value > 0) - (value < 0), cast to the value's own dtype so
            # bf16 training keeps bf16 updates (python-scalar where branches
            # would silently promote to the framework default dtype).
            positive = ops.astype(ops.greater(value, 0.0), ops.dtype(value))
            negative = ops.astype(
                ops.greater(ops.multiply(value, -1.0), 0.0), ops.dtype(value)
            )
            return ops.subtract(positive, negative)

        direction = tree_map(sign, state.m, grads)
        new_m = tree_map(
            lambda m, g: ops.add(ops.multiply(m, b2), ops.multiply(g, 1 - b2)),
            state.m,
            grads,
        )
        return direction, LionState(new_m, state.step + 1)

    return make_optimizer_chain(
        moment, initial, lr, weight_decay, decoupled_weight_decay=decoupled_weight_decay
    )


__all__ = ["lion"]
