"""RMSprop optimizer with optional DP-aware bias correction.

Hinton's RMSprop: pure second-moment EMA, no first moment::

    nu_t = α nu_{t-1} + (1 − α) g_t²
    update = g_t / (√nu_t + eps)

DP behaviour.  Under noised gradients ``g̃ = g + ξ`` with
``ξ ~ N(0, σ²)``, ``E[g̃²] = g² + σ²``, so ``nu`` is biased upward by
``σ²`` in steady state.  The vanilla optimizer survives this (the
denominator is clipped, unlike Adagrad's runaway), but the inflated
denominator shrinks the effective LR.

``NoisedPytree`` updates activate a φ-EMA correction using the realized σ
carried by the wrapper.  Unlike Adam, RMSprop does *not* divide ``nu``
by ``1 − α^t`` for bias correction, so ``φ`` and ``nu`` accumulate the
noise contribution at exactly the same rate.  Subtracting one from the
other directly yields the unbiased estimate::

    φ_t = α φ_{t-1} + (1 − α) σ_t²
    nu_corrected = max(nu_t − φ_t, floor)
    update = g_t / (√nu_corrected + eps)

``SecondMomentNoiseOutput`` substitutes a private squared-gradient ``g²``
directly into the ``nu`` update (post-processing); no φ-EMA correction is
applied in that branch.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import torch

try:
    from torchopt.base import GradientTransformation
except ImportError as exc:
    raise ImportError(
        "torchopt is required for opaque.optimizers. "
        "Install it with: pip install 'torchopt>=0.7.3'"
    ) from exc

from opaque.clipping.per_group import PerGroup
from opaque.core.pytree import tree_map
from opaque.optimizers._bias_correction import (
    is_per_group,
    resolve_noise_variance,
    update_phi_ema,
)
from opaque.optimizers._chain import make_optimizer_chain


_LR = float | Callable[[int], float]


@dataclasses.dataclass(frozen=True)
class RMSpropState:
    """State for RMSprop moment scaling.

    Attributes:
        nu: Second-moment EMA (pytree matching params).
        phi: Noise-variance EMA (scalar or ``dict[group, float]``).
            Stays at zero unless a ``NoisedPytree`` update supplies
            realized σ metadata.  Same accumulation rate as ``nu`` (no
            bias-correction division), so subtracting directly yields the
            unbiased estimate.
        step: Number of completed updates.
    """

    nu: Any
    phi: Any
    step: int


def _scale_by_rmsprop(
    alpha: float,
    eps: float,
    noise_bias_correction: bool,
    bc_floor: float,
) -> GradientTransformation:
    def init_fn(params: Any) -> RMSpropState:
        nu = tree_map(torch.zeros_like, params)
        return RMSpropState(nu=nu, phi=0.0, step=0)

    def update_fn(
        updates: Any,
        state: RMSpropState,
        *,
        params: Any = None,  # noqa: ARG001
        inplace: bool = False,  # noqa: ARG001
        noise_stddev: float | PerGroup | None = None,
        noisy_squared_grads: Any = None,
    ) -> tuple[Any, RMSpropState]:
        if noisy_squared_grads is not None and noise_stddev is not None:
            raise ValueError(
                "rmsprop.update() received both noisy_squared_grads and "
                "noise_stddev (DP-BC); pass exactly one (or neither)."
            )

        t = state.step + 1

        if noisy_squared_grads is not None:
            # External second-moment branch: g² stream replaces (g·g).
            new_nu = tree_map(
                lambda v, g2: alpha * v + (1 - alpha) * g2,
                state.nu,
                noisy_squared_grads,
            )
            new_phi = state.phi
            result = tree_map(
                lambda g, v: g / (torch.clamp(v, min=bc_floor).sqrt() + eps),
                updates,
                new_nu,
            )
            return result, RMSpropState(nu=new_nu, phi=new_phi, step=t)

        new_nu = tree_map(
            lambda v, g: alpha * v + (1 - alpha) * g * g,
            state.nu,
            updates,
        )

        effective = noise_stddev if noise_stddev is not None else 0.0
        if not noise_bias_correction:
            result = tree_map(
                lambda g, v: g / (v.sqrt() + eps),
                updates,
                new_nu,
            )
            return result, RMSpropState(nu=new_nu, phi=state.phi, step=t)

        per_group = is_per_group(effective) or isinstance(state.phi, dict)

        if per_group:
            # Per-leaf path: walk by dotted-key paths matching
            # :class:`PerGroup`'s lookup keys.
            new_phi: dict[str, float] = {}

            def _bc_walk(g_node: Any, v_node: Any, prefix: str) -> Any:
                if isinstance(g_node, dict):
                    return {
                        k: _bc_walk(
                            g_node[k],
                            v_node[k],
                            f"{prefix}.{k}" if prefix else str(k),
                        )
                        for k in g_node
                    }
                # Tensor leaf.
                path = prefix
                nv = resolve_noise_variance(effective, path)
                old_phi_k = (
                    state.phi.get(path, 0.0)
                    if isinstance(state.phi, dict)
                    else state.phi
                )
                new_phi_k = alpha * old_phi_k + (1 - alpha) * nv
                new_phi[path] = new_phi_k
                if new_phi_k > 0:
                    v_corrected = torch.clamp(v_node - new_phi_k, min=bc_floor)
                else:
                    v_corrected = v_node
                return g_node / (v_corrected.sqrt() + eps)

            result = _bc_walk(updates, new_nu, "")
        else:
            scalar_var = float(effective) ** 2
            new_phi = update_phi_ema(state.phi, scalar_var, alpha)

            if new_phi > 0:

                def _compute(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                    v_corrected = torch.clamp(v - new_phi, min=bc_floor)
                    return g / (v_corrected.sqrt() + eps)
            else:

                def _compute(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                    return g / (v.sqrt() + eps)

            result = tree_map(_compute, updates, new_nu)

        return result, RMSpropState(nu=new_nu, phi=new_phi, step=t)

    return GradientTransformation(init_fn, update_fn)


def rmsprop(
    lr: _LR = 1e-2,
    alpha: float = 0.99,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    *,
    decoupled_weight_decay: bool = True,
    update_rms_clip: float | None = None,
    noise_bias_correction: bool = False,
) -> GradientTransformation:
    """Create an RMSprop optimizer with optional DP-aware bias correction.

    Args:
        lr: Learning rate, scalar or schedule.
        alpha: Second-moment EMA decay (paper default 0.99).
        eps: Denominator stability constant.
        weight_decay: Weight-decay coefficient.
        decoupled_weight_decay: ``True`` selects decoupled WD (applied
            after moment scaling); ``False`` folds ``wd·params`` into
            the gradient before moment scaling (L2 regularisation).
        update_rms_clip: Optional StableAdamW-style RMS clip on the
            moment-scaled update.
        noise_bias_correction: If ``True``, subtract an ``alpha``-EMA of
            the realized noise variance from the second moment when
            ``NoisedPytree`` updates are passed.  Defaults to ``False``;
            flip on to ablate.

    Returns:
        A ``torchopt.base.GradientTransformation``.
    """
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if not 0 <= alpha < 1:
        raise ValueError(f"alpha must satisfy 0 <= alpha < 1, got {alpha}")
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
    if update_rms_clip is not None and update_rms_clip <= 0:
        raise ValueError(
            f"update_rms_clip must be positive when set, got {update_rms_clip}"
        )
    bc_floor = eps * eps
    moment = _scale_by_rmsprop(
        alpha=alpha,
        eps=eps,
        noise_bias_correction=noise_bias_correction,
        bc_floor=bc_floor,
    )
    return make_optimizer_chain(
        moment,
        lr=lr,
        weight_decay=weight_decay,
        decoupled_weight_decay=decoupled_weight_decay,
        update_rms_clip=update_rms_clip,
    )


__all__ = ["rmsprop", "RMSpropState"]
