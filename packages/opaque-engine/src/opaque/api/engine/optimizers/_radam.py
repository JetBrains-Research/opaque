"""Backend-neutral Rectified Adam."""

from __future__ import annotations

import math
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
from opaque.api.engine.optimizers.types import RAdamState
from opaque.pytree import tree_map

_LR = float | Callable[[int], float]


def _rho_t(b2: float, t: int) -> float:
    """Return RAdam's rectification length at a positive update step."""
    rho_inf = 2.0 / (1.0 - b2) - 1.0
    return rho_inf - 2.0 * t * b2**t / (1.0 - b2**t)


def _rectification(b2: float, t: int) -> float | None:
    """Return the RAdam scale, or ``None`` during the SGD warmup phase."""
    rho = _rho_t(b2, t)
    if rho <= 5.0:
        return None
    rho_inf = 2.0 / (1.0 - b2) - 1.0
    return math.sqrt(
        (rho - 4.0) * (rho - 2.0) * rho_inf / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho)
    )


def radam(
    params: Any,
    lr: _LR = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    *,
    decoupled_weight_decay: bool = False,
    update_rms_clip: float | None = None,
    noise_bias_correction: bool = False,
) -> tuple[Callable[..., tuple[Any, RAdamState]], RAdamState]:
    if (
        len(betas) != 2
        or not all(0 <= beta < 1 for beta in betas)
        or eps <= 0
        or weight_decay < 0
    ):
        raise ValueError("invalid RAdam hyperparameters")
    if update_rms_clip is not None and update_rms_clip <= 0:
        raise ValueError("update_rms_clip must be positive")
    b1, b2 = betas
    initial = RAdamState(
        tree_map(ops.zeros_like, params),
        tree_map(ops.zeros_like, params),
        _init_per_group_phi(params) if noise_bias_correction else 0.0,
        0,
    )

    def moment(
        grads: Any, state: RAdamState, _params: Any, noise_stddev: Any, squared: Any
    ) -> tuple[Any, RAdamState]:
        t = state.step + 1
        bc1, bc2 = 1 - b1**t, 1 - b2**t
        rho = _rho_t(b2, t)
        mu = tree_map(
            lambda m, g: ops.add(ops.multiply(m, b1), ops.multiply(g, 1 - b1)),
            state.mu,
            grads,
        )
        nu = tree_map(
            lambda v, g2: ops.add(ops.multiply(v, b2), ops.multiply(g2, 1 - b2)),
            state.nu,
            squared if squared is not None else tree_map(ops.square, grads),
        )
        if squared is not None or not noise_bias_correction:
            phi = state.phi
        else:
            effective = noise_stddev if noise_stddev is not None else 0.0
            if _is_per_group(effective) or isinstance(state.phi, dict):
                phi = {
                    path: b2
                    * (
                        state.phi.get(path, 0.0)
                        if isinstance(state.phi, dict)
                        else float(state.phi)
                    )
                    + (1 - b2) * _resolve_noise_variance(effective, path)
                    for path in _init_per_group_phi(mu)
                }
            else:
                phi = b2 * float(state.phi) + (1 - b2) * float(effective) ** 2
        if rho <= 5:
            return tree_map(lambda m: ops.divide(m, bc1), mu), RAdamState(
                mu, nu, phi, t
            )
        r = _rectification(b2, t)
        assert r is not None

        def compute(m: Any, v: Any, correction: float = 0.0) -> Any:
            v_raw = ops.divide(v, bc2)
            candidate = ops.subtract(v_raw, correction) if correction > 0 else v_raw
            v_eff = ops.where(ops.greater(candidate, 0.0), candidate, v_raw)
            return ops.multiply(
                ops.divide(
                    ops.divide(m, bc1),
                    ops.add(ops.sqrt(ops.clamp(v_eff, eps * eps)), eps),
                ),
                r,
            )

        if isinstance(phi, dict) and squared is None and noise_bias_correction:
            result = _map_leaves_with_path(
                lambda path, m, v: compute(m, v, phi[path] / bc2), mu, nu
            )
        else:
            correction = (
                0.0
                if squared is not None or not noise_bias_correction
                else float(phi) / bc2
            )
            result = tree_map(lambda m, v: compute(m, v, correction), mu, nu)
        return result, RAdamState(mu, nu, phi, t)

    return make_optimizer_chain(
        moment,
        initial,
        lr,
        weight_decay,
        decoupled_weight_decay=decoupled_weight_decay,
        update_rms_clip=update_rms_clip,
    )


__all__ = ["_rectification", "_rho_t", "radam"]
