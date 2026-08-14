"""Backend-neutral factored Adafactor."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from opaque.api.engine import ops
from opaque.api.engine.optimizers._bias_correction import resolve_noise_variance
from opaque.api.engine.optimizers._chain import (
    _clip_by_global_rms,
    make_optimizer_chain,
)
from opaque.api.engine.optimizers.types import AdafactorState
from opaque.api.engine.pytree import tree_flatten_with_paths, tree_unflatten
from opaque.pytree import tree_map

_LR = float | Callable[[int], float]


def _initial_v(value: Any) -> tuple[Any, ...]:
    shape = ops.shape(value)
    if len(shape) >= 2:
        return ops.zeros(shape[:-1], dtype=ops.dtype(value), like=value), ops.zeros(
            shape[:-2] + shape[-1:], dtype=ops.dtype(value), like=value
        )
    return (ops.zeros_like(value),)


def _v_hat(value: tuple[Any, ...], eps_root: float) -> Any:
    if len(value) == 1:
        return value[0]
    row, col = value
    row_mean = ops.mean(row, axis=-1)
    scale = ops.maximum(ops.mean(row), ops.multiply(ops.mean(row), eps_root))
    row_mean = ops.maximum(row_mean, scale)
    return ops.multiply(
        ops.expand_dims(ops.divide(row, row_mean), -1), ops.expand_dims(col, -2)
    )


def adafactor(
    params: Any,
    lr: _LR = 1e-3,
    beta1: float = 0.0,
    decay_rate: float = -0.8,
    eps_grad: float = 1e-30,
    eps_root: float = 1e-3,
    weight_decay: float = 0.0,
    update_rms_clip: float = 1.0,
    *,
    decoupled_weight_decay: bool = True,
    noise_bias_correction: bool = False,
) -> tuple[Callable[..., tuple[Any, AdafactorState]], AdafactorState]:
    if (
        not 0 <= beta1 < 1
        or decay_rate >= 0
        or eps_grad <= 0
        or eps_root <= 0
        or weight_decay < 0
        or update_rms_clip <= 0
    ):
        raise ValueError("invalid Adafactor hyperparameters")
    paths, leaves, spec = tree_flatten_with_paths(params)
    initial = AdafactorState(
        tree_map(ops.zeros_like, params) if beta1 else None,
        tuple(_initial_v(leaf) for leaf in leaves),
        tuple(0.0 for _ in leaves),
        spec,
        tuple(paths),
        0,
    )

    def moment(
        grads: Any, state: AdafactorState, _params: Any, noise_stddev: Any, squared: Any
    ) -> tuple[Any, AdafactorState]:
        if squared is not None:
            raise ValueError("Adafactor does not support private second-moment streams")
        paths_, grads, _spec = tree_flatten_with_paths(grads)
        if tuple(paths_) != state.paths:
            raise ValueError("updates pytree does not match optimizer state")
        t, beta2 = state.step + 1, 1 - (state.step + 1) ** decay_rate
        next_v, next_phi, directions = [], [], []
        for path, grad, old_v, old_phi in zip(
            state.paths, grads, state.v_flat, state.phi_flat, strict=True
        ):
            g2 = ops.square(grad)
            if len(old_v) == 2:
                row, col = old_v
                value = (
                    ops.add(
                        ops.multiply(row, beta2),
                        ops.multiply(ops.mean(g2, axis=-1), 1 - beta2),
                    ),
                    ops.add(
                        ops.multiply(col, beta2),
                        ops.multiply(ops.mean(g2, axis=-2), 1 - beta2),
                    ),
                )
            else:
                value = (
                    ops.add(ops.multiply(old_v[0], beta2), ops.multiply(g2, 1 - beta2)),
                )
            phi = beta2 * old_phi + (1 - beta2) * (
                resolve_noise_variance(noise_stddev, path)
                if noise_bias_correction and noise_stddev is not None
                else 0.0
            )
            v = _v_hat(value, eps_root)
            if noise_bias_correction and phi > 0:
                corrected = ops.subtract(v, phi)
                v = ops.where(ops.greater(corrected, 0.0), corrected, v)
            directions.append(
                ops.divide(grad, ops.add(ops.sqrt(ops.clamp(v, eps_grad)), eps_root))
            )
            next_v.append(value)
            next_phi.append(phi)
        direction = tree_unflatten(state.treespec, directions)
        direction = _clip_by_global_rms(direction, update_rms_clip)
        if beta1:
            new_m = tree_map(
                lambda m, d: ops.add(
                    ops.multiply(m, beta1), ops.multiply(d, 1 - beta1)
                ),
                state.m,
                direction,
            )
            direction = new_m
        else:
            new_m = None
        return direction, AdafactorState(
            new_m, tuple(next_v), tuple(next_phi), state.treespec, state.paths, t
        )

    return make_optimizer_chain(
        moment, initial, lr, weight_decay, decoupled_weight_decay=decoupled_weight_decay
    )


__all__ = ["adafactor"]
