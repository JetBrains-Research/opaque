"""Backend-neutral Adadelta."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from opaque.api.engine import ops
from opaque.api.engine.optimizers._bias_correction import (
    init_per_group_phi,
    is_per_group,
    map_leaves_with_path,
    resolve_noise_variance,
)
from opaque.api.engine.optimizers._chain import make_optimizer_chain
from opaque.api.engine.optimizers.types import AdadeltaState
from opaque.pytree import tree_map

_LR = float | Callable[[int], float]


def adadelta(
    params: Any,
    lr: _LR = 1.0,
    rho: float = 0.9,
    eps: float = 1e-6,
    weight_decay: float = 0.0,
    *,
    decoupled_weight_decay: bool = True,
    update_rms_clip: float | None = None,
    noise_bias_correction: bool = False,
) -> tuple[Callable[..., tuple[Any, AdadeltaState]], AdadeltaState]:
    if (
        not 0 <= rho < 1
        or eps <= 0
        or weight_decay < 0
        or (update_rms_clip is not None and update_rms_clip <= 0)
    ):
        raise ValueError("invalid Adadelta hyperparameters")
    zeros = tree_map(ops.zeros_like, params)
    initial = AdadeltaState(
        zeros,
        tree_map(ops.zeros_like, params),
        init_per_group_phi(params) if noise_bias_correction else None,
        tree_map(ops.zeros_like, params) if noise_bias_correction else None,
        0,
    )

    def moment(
        grads: Any, state: AdadeltaState, _params: Any, noise_stddev: Any, squared: Any
    ) -> tuple[Any, AdadeltaState]:
        vg = tree_map(
            lambda v, g2: ops.add(ops.multiply(v, rho), ops.multiply(g2, 1 - rho)),
            state.v_g,
            squared if squared is not None else tree_map(ops.square, grads),
        )
        effective = noise_stddev if noise_stddev is not None else 0.0
        if noise_bias_correction and squared is None:
            if is_per_group(effective) or isinstance(state.phi_g, dict):
                phi_g = {
                    path: rho
                    * (
                        state.phi_g.get(path, 0.0)
                        if isinstance(state.phi_g, dict)
                        else 0.0
                    )
                    + (1 - rho) * resolve_noise_variance(effective, path)
                    for path in init_per_group_phi(vg)
                }
            else:
                phi_g = (
                    rho * float(state.phi_g or 0.0) + (1 - rho) * float(effective) ** 2
                )
        else:
            phi_g = state.phi_g

        def coefficient(
            g: Any,
            grad_var: Any,
            update_var: Any,
            phi_g_value: float,
            phi_dx_node: Any = None,
        ) -> Any:
            corrected_g = (
                ops.subtract(grad_var, phi_g_value) if phi_g_value > 0 else grad_var
            )
            g_eff = ops.where(
                ops.greater(corrected_g, 0.0), corrected_g, ops.clamp(grad_var, 0.0)
            )
            if phi_dx_node is not None:
                # Two-EMA DP bias correction applies to the numerator too:
                # subtract the update-noise EMA from E[dx**2], falling back
                # to the uncorrected value when noise dominates.
                corrected_dx = ops.subtract(update_var, phi_dx_node)
                dx_eff = ops.where(
                    ops.greater(corrected_dx, 0.0), corrected_dx, update_var
                )
            else:
                dx_eff = update_var
            return ops.divide(
                ops.sqrt(ops.add(dx_eff, eps)), ops.sqrt(ops.add(g_eff, eps))
            )

        bc_dx_active = (
            noise_bias_correction and squared is None and state.phi_dx is not None
        )
        if (
            noise_bias_correction
            and squared is None
            and (is_per_group(effective) or isinstance(phi_g, dict))
        ):
            if bc_dx_active:
                coeff = map_leaves_with_path(
                    lambda path, g, v, dx, pdx: coefficient(g, v, dx, phi_g[path], pdx),
                    grads,
                    vg,
                    state.v_dx,
                    state.phi_dx,
                )
            else:
                coeff = map_leaves_with_path(
                    lambda path, g, v, dx: coefficient(g, v, dx, phi_g[path]),
                    grads,
                    vg,
                    state.v_dx,
                )
        else:
            scalar_phi = float(phi_g or 0.0) if not isinstance(phi_g, dict) else 0.0
            if bc_dx_active:
                coeff = tree_map(
                    lambda g, v, dx, pdx: coefficient(g, v, dx, scalar_phi, pdx),
                    grads,
                    vg,
                    state.v_dx,
                    state.phi_dx,
                )
            else:
                coeff = tree_map(
                    lambda g, v, dx: coefficient(g, v, dx, scalar_phi),
                    grads,
                    vg,
                    state.v_dx,
                )
        update = tree_map(ops.multiply, coeff, grads)
        vdx = tree_map(
            lambda v, u: ops.add(
                ops.multiply(v, rho), ops.multiply(ops.square(u), 1 - rho)
            ),
            state.v_dx,
            update,
        )
        if noise_bias_correction and squared is None:
            if state.phi_dx is None:
                raise ValueError("Adadelta checkpoint is missing phi_dx")
            if is_per_group(effective):
                phi_dx = map_leaves_with_path(
                    lambda path, old, c: ops.add(
                        ops.multiply(old, rho),
                        ops.multiply(
                            ops.square(c),
                            (1 - rho) * resolve_noise_variance(effective, path),
                        ),
                    ),
                    state.phi_dx,
                    coeff,
                )
            else:
                phi_dx = tree_map(
                    lambda old, c: ops.add(
                        ops.multiply(old, rho),
                        ops.multiply(ops.square(c), (1 - rho) * float(effective) ** 2),
                    ),
                    state.phi_dx,
                    coeff,
                )
        else:
            phi_dx = state.phi_dx
        return update, AdadeltaState(vg, vdx, phi_g, phi_dx, state.step + 1)

    return make_optimizer_chain(
        moment,
        initial,
        lr,
        weight_decay,
        decoupled_weight_decay=decoupled_weight_decay,
        update_rms_clip=update_rms_clip,
    )


__all__ = ["adadelta"]
