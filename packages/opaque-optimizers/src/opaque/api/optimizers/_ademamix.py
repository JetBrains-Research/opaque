"""Backend-neutral AdEMAMix optimizer."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from opaque.api.engine import ops
from opaque.api.optimizers._adam import (
    _init_per_group_phi,
    _is_per_group,
    _map_leaves_with_path,
    _resolve_noise_variance,
)
from opaque.api.optimizers._chain import make_optimizer_chain
from opaque.api.optimizers.types import AdEMAMixState
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
    if len(betas) != 3:
        raise ValueError(f"betas must contain exactly three values, got {betas}")
    for _name, _beta in zip(("β₁", "β₂", "β₃"), betas, strict=True):
        if not 0 <= _beta < 1:
            raise ValueError(f"{_name} must satisfy 0 <= b < 1, got {_beta}")
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
    if update_rms_clip is not None and update_rms_clip <= 0:
        raise ValueError(
            f"update_rms_clip must be positive when set, got {update_rms_clip}"
        )
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

        def compute_squared_stream(mf_: Any, ms_: Any, v: Any) -> Any:
            # Private second-moment streams can drive v negative; fall back
            # to the combined moment's square and floor the denominator.
            combined = ops.add(ops.divide(mf_, bc1), ops.multiply(ms_, alpha))
            v_hat = ops.divide(v, bc2)
            v_eff = ops.clamp(
                ops.where(ops.greater(v_hat, 0.0), v_hat, ops.square(combined)),
                eps * eps,
            )
            return ops.divide(combined, ops.add(ops.sqrt(v_eff), eps))

        def compute(mf_: Any, ms_: Any, v: Any, phi: float = 0.0) -> Any:
            m_hat = ops.add(ops.divide(mf_, bc1), ops.multiply(ms_, alpha))
            v_raw = ops.divide(v, bc2)
            if phi > 0:
                # DP bias correction: subtract the noise-variance EMA, falling
                # back to the *uncorrected* second moment (not a proxy) when
                # noise dominates; no extra floor beyond eps in the sqrt sum.
                corrected = ops.subtract(v_raw, phi)
                v_eff = ops.where(ops.greater(corrected, 0.0), corrected, v_raw)
            else:
                v_eff = v_raw
            return ops.divide(m_hat, ops.add(ops.sqrt(v_eff), eps))

        if squared is not None:
            result = tree_map(compute_squared_stream, mf, ms, nu)
            new_phi = state.phi
        elif not noise_bias_correction:
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
