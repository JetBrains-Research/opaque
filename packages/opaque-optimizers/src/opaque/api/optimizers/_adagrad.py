"""Backend-neutral Adagrad with cumulative DP variance subtraction."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from opaque.api.engine import ops
from opaque.api.optimizers._bias_correction import (
    init_per_group_phi,
    is_per_group,
    map_leaves_with_path,
    resolve_noise_variance,
)
from opaque.api.optimizers._chain import make_optimizer_chain
from opaque.api.optimizers.types import AdagradState
from opaque.pytree import tree_map

_LR = float | Callable[[int], float]


def adagrad(
    params: Any,
    lr: _LR = 1e-2,
    eps: float = 1e-10,
    weight_decay: float = 0.0,
    initial_accumulator_value: float = 0.0,
    *,
    decoupled_weight_decay: bool = True,
    noise_bias_correction: bool = False,
) -> tuple[Callable[..., tuple[Any, AdagradState]], AdagradState]:
    if eps <= 0 or weight_decay < 0 or initial_accumulator_value < 0:
        raise ValueError("invalid Adagrad hyperparameters")
    initial = AdagradState(
        tree_map(
            lambda p: ops.add(ops.zeros_like(p), initial_accumulator_value), params
        ),
        init_per_group_phi(params) if noise_bias_correction else 0.0,
        0,
    )

    def moment(
        grads: Any, state: AdagradState, _params: Any, noise_stddev: Any, _squared: Any
    ) -> tuple[Any, AdagradState]:
        new_v = tree_map(lambda v, g: ops.add(v, ops.square(g)), state.v_acc, grads)
        if not noise_bias_correction:
            return tree_map(
                lambda g, v: ops.divide(g, ops.add(ops.sqrt(v), eps)), grads, new_v
            ), AdagradState(new_v, state.phi_acc, state.step + 1)
        effective = noise_stddev if noise_stddev is not None else 0.0
        if is_per_group(effective) or isinstance(state.phi_acc, dict):
            new_phi: dict[Any, float] = {}

            def compute(path: Any, g: Any, v: Any) -> Any:
                phi = (
                    state.phi_acc.get(path, 0.0)
                    if isinstance(state.phi_acc, dict)
                    else float(state.phi_acc)
                ) + resolve_noise_variance(effective, path)
                new_phi[path] = phi
                corrected = ops.subtract(v, phi) if phi > 0 else v
                return ops.divide(
                    g,
                    ops.add(
                        ops.sqrt(ops.where(ops.greater(corrected, 0.0), corrected, v)),
                        eps,
                    ),
                )

            result = map_leaves_with_path(compute, grads, new_v)
        else:
            new_phi = float(state.phi_acc) + float(effective) ** 2
            result = tree_map(
                lambda g, v: ops.divide(
                    g,
                    ops.add(
                        ops.sqrt(
                            ops.where(
                                ops.greater(ops.subtract(v, new_phi), 0.0),
                                ops.subtract(v, new_phi),
                                v,
                            )
                        ),
                        eps,
                    ),
                ),
                grads,
                new_v,
            )
        return result, AdagradState(new_v, new_phi, state.step + 1)

    return make_optimizer_chain(
        moment, initial, lr, weight_decay, decoupled_weight_decay=decoupled_weight_decay
    )


__all__ = ["adagrad"]
