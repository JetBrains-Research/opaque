"""Backend-neutral SGD optimizer factory."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from opaque.api.engine import ops
from opaque.api.optimizers._chain import make_optimizer_chain
from opaque.api.optimizers.types import SGDState
from opaque.pytree import tree_map

_LR = float | Callable[[int], float]


def _sgd_step(
    momentum: float,
    dampening: float,
    nesterov: bool,
) -> Callable[..., tuple[Any, SGDState]]:
    """SGD momentum primitive.

    On the first update the buffer is initialised to the incoming gradient,
    ignoring ``dampening``. Subsequent updates use the standard momentum update.
    """

    def step(
        updates: Any,
        state: SGDState,
        params: Any,
        noise_stddev: float | Any | None = None,
        noisy_squared_grads: Any | None = None,
    ) -> tuple[Any, SGDState]:
        del params, noise_stddev, noisy_squared_grads

        t = state.step + 1
        first_call = state.step == 0

        if momentum == 0.0:
            new_buffers = None
            output = updates
        else:
            if state.momentum is None:
                buffers = tree_map(lambda g: ops.zeros_like(g), updates)
            else:
                buffers = state.momentum

            if first_call:
                new_buffers = tree_map(
                    lambda _b, g: ops.clone(g),
                    buffers,
                    updates,
                )
            else:
                new_buffers = tree_map(
                    lambda b, g: ops.add(
                        ops.multiply(b, momentum),
                        ops.multiply(g, 1.0 - dampening),
                    ),
                    buffers,
                    updates,
                )

            if nesterov:
                output = tree_map(
                    lambda g, b: ops.add(g, ops.multiply(b, momentum)),
                    updates,
                    new_buffers,
                )
            else:
                output = tree_map(lambda b: ops.clone(b), new_buffers)

        return output, SGDState(momentum=new_buffers, step=t)

    return step


def sgd(
    params: Any,
    lr: _LR,
    momentum: float = 0.0,
    dampening: float = 0.0,
    weight_decay: float = 0.0,
    nesterov: bool = False,
    *,
    maximize: bool = False,
) -> tuple[Callable[..., tuple[Any, SGDState]], SGDState]:
    """Create SGD with Opaque's wrapper-aware update API.

    SGD's update is unbiased under additive zero-mean DP noise, so it does
    not consume ``NoisedPytree.noise_stddev``. The wrapper accepts
    ``NoisedPytree`` anyway and forwards the privatized pytree to the
    internal momentum logic.

    Args:
        params: Parameter pytree used to initialise optimizer state.
        lr: Learning rate, scalar or ``step -> float`` callable schedule.
        momentum: Momentum factor.
        dampening: Dampening for momentum.
        weight_decay: L2 weight-decay coefficient.
        nesterov: Use Nesterov momentum.
        maximize: Maximize the objective instead of minimizing.

    Returns:
        ``(step_fn, SGDState)``.
    """
    if momentum < 0:
        raise ValueError(f"momentum must be non-negative, got {momentum}")
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
    if nesterov and (momentum <= 0.0 or dampening != 0.0):
        raise ValueError("Nesterov momentum requires a momentum and zero dampening")

    init_state = SGDState(
        momentum=None
        if momentum == 0.0
        else tree_map(lambda p: ops.zeros_like(p), params),
        step=0,
    )

    moment_step = _sgd_step(
        momentum=momentum,
        dampening=dampening,
        nesterov=nesterov,
    )

    return make_optimizer_chain(
        moment_step=moment_step,
        moment_init_state=init_state,
        lr=lr,
        weight_decay=weight_decay,
        decoupled_weight_decay=False,
        maximize=maximize,
    )


__all__ = ["sgd"]
