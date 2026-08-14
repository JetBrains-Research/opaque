"""Backend-neutral AdEMAMix optimizer."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from opaque.api.engine import ops
from opaque.api.engine.optimizers._adam import (
    _init_per_group_phi,
    _is_per_group,
    _map_leaves_with_path,
    _resolve_noise_variance,
)
from opaque.api.engine.optimizers._chain import make_optimizer_chain
from opaque.api.engine.optimizers.types import AdEMAMixState
from opaque.pytree import tree_map

_LR = float | Callable[[int], float]


def ademamix(
    params: Any,
    lr: _LR = 1e-3,
    betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
    alpha: float = 5.0,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    *,
    decoupled_weight_decay: bool = True,
    update_rms_clip: float | None = None,
    noise_bias_correction: bool = False,
) -> tuple[Callable[..., tuple[Any, AdEMAMixState]], AdEMAMixState]:
    if (
        len(betas) != 3
        or not all(0 <= beta < 1 for beta in betas)
        or alpha < 0
        or eps <= 0
        or weight_decay < 0
    ):
        raise ValueError("invalid AdEMAMix hyperparameters")
    if update_rms_clip is not None and update_rms_clip <= 0:
        raise ValueError("update_rms_clip must be positive")
    b1, b2, b3 = betas
    zeros = tree_map(ops.zeros_like, params)
    initial = AdEMAMixState(
        zeros,
        tree_map(ops.zeros_like, params),
        tree_map(ops.zeros_like, params),
        _init_per_group_phi(params) if noise_bias_correction else 0.0,
        0,
    )

    def moment(
        grads: Any, state: AdEMAMixState, _params: Any, noise_stddev: Any, squared: Any
    ) -> tuple[Any, AdEMAMixState]:
        t, bc1, bc2 = (
            state.step + 1,
            1 - b1 ** (state.step + 1),
            1 - b2 ** (state.step + 1),
        )
        mf = tree_map(
            lambda m, g: ops.add(ops.multiply(m, b1), ops.multiply(g, 1 - b1)),
            state.m_fast,
            grads,
        )
        ms = tree_map(
            lambda m, g: ops.add(ops.multiply(m, b3), ops.multiply(g, 1 - b3)),
            state.m_slow,
            grads,
        )
        nu = tree_map(
            lambda v, g2: ops.add(ops.multiply(v, b2), ops.multiply(g2, 1 - b2)),
            state.nu,
            squared if squared is not None else tree_map(ops.square, grads),
        )

        def compute(mf_: Any, ms_: Any, v: Any, phi: float = 0.0) -> Any:
            m_hat = ops.add(ops.divide(mf_, bc1), ops.multiply(ms_, alpha))
            v_raw = ops.divide(v, bc2)
            corrected = ops.subtract(v_raw, phi) if phi > 0 else v_raw
            # Private second-moment streams can be negative; fall back to a
            # non-negative proxy and floor the denominator, matching Adam.
            v_eff = ops.where(
                ops.greater(corrected, 0.0),
                corrected,
                ops.square(m_hat),
            )
            v_eff = ops.clamp(v_eff, lo=eps * eps)
            return ops.divide(m_hat, ops.add(ops.sqrt(v_eff), eps))

        if squared is not None or not noise_bias_correction:
            result = tree_map(lambda a, b, v: compute(a, b, v), mf, ms, nu)
            new_phi = state.phi
        else:
            effective = noise_stddev if noise_stddev is not None else 0.0
            if _is_per_group(effective) or isinstance(state.phi, dict):
                new_phi: dict[Any, float] = {}

                def per_leaf(path: Any, a: Any, b: Any, v: Any) -> Any:
                    phi = b2 * (
                        state.phi.get(path, 0.0)
                        if isinstance(state.phi, dict)
                        else float(state.phi)
                    ) + (1 - b2) * _resolve_noise_variance(effective, path)
                    new_phi[path] = phi
                    return compute(a, b, v, phi / bc2)

                result = _map_leaves_with_path(per_leaf, mf, ms, nu)
            else:
                new_phi = b2 * float(state.phi) + (1 - b2) * float(effective) ** 2
                result = tree_map(
                    lambda a, b, v: compute(a, b, v, new_phi / bc2), mf, ms, nu
                )
        return result, AdEMAMixState(mf, ms, nu, new_phi, t)

    return make_optimizer_chain(
        moment,
        initial,
        lr,
        weight_decay,
        decoupled_weight_decay=decoupled_weight_decay,
        update_rms_clip=update_rms_clip,
    )


__all__ = ["ademamix"]
