"""Backend-neutral RMSprop with DP variance correction."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from opaque.api.engine import ops
from opaque.api.optimizers._bias_correction import (
    init_per_group_phi,
    is_per_group,
    map_leaves_with_path,
    resolve_noise_variance,
    update_phi_ema,
)
from opaque.api.optimizers._chain import make_optimizer_chain
from opaque.api.optimizers.types import RMSpropState
from opaque.pytree import tree_map

_LR = float | Callable[[int], float]


def rmsprop(
    params: Any,
    lr: _LR = 1e-2,
    alpha: float = 0.99,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    *,
    decoupled_weight_decay: bool = True,
    update_rms_clip: float | None = None,
    noise_bias_correction: bool = False,
) -> tuple[Callable[..., tuple[Any, RMSpropState]], RMSpropState]:
    if (
        not 0 <= alpha < 1
        or eps <= 0
        or weight_decay < 0
        or (update_rms_clip is not None and update_rms_clip <= 0)
    ):
        raise ValueError("invalid RMSprop hyperparameters")
    initial = RMSpropState(
        tree_map(ops.zeros_like, params),
        init_per_group_phi(params) if noise_bias_correction else 0.0,
        0,
    )

    def moment(
        grads: Any, state: RMSpropState, _params: Any, noise_stddev: Any, squared: Any
    ) -> tuple[Any, RMSpropState]:
        source = squared if squared is not None else tree_map(ops.square, grads)
        new_nu = tree_map(
            lambda v, g2: ops.add(ops.multiply(v, alpha), ops.multiply(g2, 1 - alpha)),
            state.nu,
            source,
        )
        if squared is not None:

            def _compute_sm(g: Any, v: Any) -> Any:
                # Noisy g**2 can be negative when second-stream noise
                # dominates the signal; fall back to g**2 so the update
                # magnitude stays ~1 instead of exploding off the floor.
                v_eff = ops.clamp(
                    ops.where(ops.greater(v, 0.0), v, ops.square(g)), eps * eps
                )
                return ops.divide(g, ops.add(ops.sqrt(v_eff), eps))

            result = tree_map(_compute_sm, grads, new_nu)
            return result, RMSpropState(new_nu, state.phi, state.step + 1)
        if not noise_bias_correction:
            result = tree_map(
                lambda g, v: ops.divide(g, ops.add(ops.sqrt(v), eps)), grads, new_nu
            )
            return result, RMSpropState(new_nu, state.phi, state.step + 1)
        effective = noise_stddev if noise_stddev is not None else 0.0
        if is_per_group(effective) or isinstance(state.phi, dict):
            new_phi: dict[Any, float] = {}

            def compute(path: Any, g: Any, v: Any) -> Any:
                phi = alpha * (
                    state.phi.get(path, 0.0)
                    if isinstance(state.phi, dict)
                    else float(state.phi)
                ) + (1 - alpha) * resolve_noise_variance(effective, path)
                new_phi[path] = phi
                corrected = ops.subtract(v, phi) if phi > 0 else v
                return ops.divide(
                    g,
                    ops.add(
                        ops.sqrt(ops.where(ops.greater(corrected, 0.0), corrected, v)),
                        eps,
                    ),
                )

            result = map_leaves_with_path(compute, grads, new_nu)
        else:
            new_phi = update_phi_ema(state.phi, float(effective) ** 2, alpha)
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
                new_nu,
            )
        return result, RMSpropState(new_nu, new_phi, state.step + 1)

    return make_optimizer_chain(
        moment,
        initial,
        lr,
        weight_decay,
        decoupled_weight_decay=decoupled_weight_decay,
        update_rms_clip=update_rms_clip,
    )


__all__ = ["rmsprop"]
