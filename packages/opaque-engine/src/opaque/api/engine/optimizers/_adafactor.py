"""Backend-neutral factored Adafactor (Shazeer & Stern, 2018).

Memory-efficient Adam variant: the second moment is **factored** for
tensors of rank >= 2 (one ``v_row`` over the ``-2`` axis and one
``v_col`` over the ``-1`` axis instead of a full matrix; higher-rank
tensors collapse their leading dims into rows). Tensors of rank < 2
fall back to an elementwise second moment.

The factored estimator approximates the full second moment as
``v_row[..., :, None] * v_col[..., None, :] / mean(v_row, axis=-1)``
(the paper's Algorithm 4). Both stability floors are **scale-relative**
— ``eps_root`` applies as a fraction of the factor's own mean — so they
do not fire spuriously on the small gradients DP clipping produces
while still guarding genuine numerical underflow.

DP noise-variance bias correction: under additive Gaussian noise the
bias on ``E[g**2]`` is a uniform ``+sigma**2`` per element, which
propagates cleanly through the row and column means, so a single
``beta2_t``-weighted phi EMA is subtracted from **each factor** (with a
positive-part guard) before the factored approximation is composed.
The private second-moment substitution path is not offered: substituting
a privately-estimated ``g**2`` stream does not preserve the rank-1
factorisation in any obvious way.
"""

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

_UNDERFLOW_FLOOR = 1e-30


def _initial_v(value: Any) -> tuple[Any, ...]:
    shape = ops.shape(value)
    if len(shape) >= 2:
        return (
            ops.zeros(shape[:-1], dtype=ops.dtype(value), like=value),
            ops.zeros(shape[:-2] + shape[-1:], dtype=ops.dtype(value), like=value),
        )
    return (ops.zeros_like(value),)


def _mean_keepdim(value: Any, axis: int) -> Any:
    return ops.expand_dims(ops.mean(value, axis=axis), axis)


def _approx_v_hat(v_row: Any, v_col: Any, eps_root: float) -> Any:
    """Factored ``v_hat`` with a scale-relative floor on the row mean."""
    r_mean = _mean_keepdim(v_row, -1)
    v_row_scale = ops.clamp(ops.mean(v_row), _UNDERFLOW_FLOOR)
    r_mean = ops.maximum(r_mean, ops.multiply(v_row_scale, eps_root))
    return ops.multiply(
        ops.expand_dims(ops.divide(v_row, r_mean), -1),
        ops.expand_dims(v_col, -2),
    )


def _floored_rsqrt_scale(grad: Any, v_eff: Any, eps_root: float) -> Any:
    """``grad / sqrt(v_eff)`` with a scale-relative denominator floor."""
    v_scale = ops.sqrt(ops.clamp(ops.mean(v_eff), _UNDERFLOW_FLOOR))
    denominator = ops.maximum(ops.sqrt(v_eff), ops.multiply(v_scale, eps_root))
    return ops.divide(grad, denominator)


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
        t = state.step + 1
        beta2 = 1.0 - float(t) ** decay_rate
        bc_active = noise_bias_correction and noise_stddev is not None

        next_v, next_phi, directions = [], [], []
        for path, grad, old_v, old_phi in zip(
            state.paths, grads, state.v_flat, state.phi_flat, strict=True
        ):
            g_sq = ops.add(ops.square(grad), eps_grad)

            if bc_active:
                phi = beta2 * old_phi + (1.0 - beta2) * resolve_noise_variance(
                    noise_stddev, path
                )
            else:
                phi = old_phi
            next_phi.append(phi)

            if len(old_v) == 2:
                row, col = old_v
                new_row = ops.add(
                    ops.multiply(row, beta2),
                    ops.multiply(ops.mean(g_sq, axis=-1), 1.0 - beta2),
                )
                new_col = ops.add(
                    ops.multiply(col, beta2),
                    ops.multiply(ops.mean(g_sq, axis=-2), 1.0 - beta2),
                )
                if bc_active and phi > 0.0:
                    corr_row = ops.subtract(new_row, phi)
                    corr_col = ops.subtract(new_col, phi)
                    row_eff = ops.where(ops.greater(corr_row, 0.0), corr_row, new_row)
                    col_eff = ops.where(ops.greater(corr_col, 0.0), corr_col, new_col)
                else:
                    row_eff, col_eff = new_row, new_col
                v_hat = _approx_v_hat(row_eff, col_eff, eps_root)
                directions.append(_floored_rsqrt_scale(grad, v_hat, eps_root))
                next_v.append((new_row, new_col))
            else:
                (v,) = old_v
                new_v = ops.add(ops.multiply(v, beta2), ops.multiply(g_sq, 1.0 - beta2))
                if bc_active and phi > 0.0:
                    corr = ops.subtract(new_v, phi)
                    v_eff = ops.where(ops.greater(corr, 0.0), corr, new_v)
                else:
                    v_eff = new_v
                directions.append(_floored_rsqrt_scale(grad, v_eff, eps_root))
                next_v.append((new_v,))

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
