"""AdEMAMix optimizer (Pagliardini et al., 2024).

Adam variant with **two** first-moment EMAs (a "fast" β₁ and a "slow"
β₃) blended by ``α``::

    m_fast_t = β₁ m_fast_{t-1} + (1 − β₁) g_t
    m_slow_t = β₃ m_slow_{t-1} + (1 − β₃) g_t        # no bias correction (β₃ ≈ 0.9999)
    v_t      = β₂ v_{t-1}      + (1 − β₂) g_t²

    m̂_fast = m_fast_t / (1 − β₁^t)
    v̂      = v_t      / (1 − β₂^t)

    update  = (m̂_fast + α · m_slow_t) / (√v̂ + ε)

Reference:
    Pagliardini, Ablin, Grangier, "The AdEMAMix Optimizer: Better,
    Faster, Older", arXiv:2409.03137.

The slow EMA is intentionally **not** bias-corrected (the paper notes
that with β₃ on the order of 0.9999 the bias is negligible after a
brief warm-up, and applying ``1 − β₃^t`` makes early steps numerically
unstable).

DP behavior.  The second moment EMA is structurally identical to
Adam's, so:

- ``noise_stddev`` (default + per-step override) drives the φ-EMA bias
  correction on ``v̂`` exactly as in :func:`opaque.optimizers.adamw`.
- ``noisy_squared_grads`` substitutes the JME paired-stream second
  moment by post-processing.
- The two first-moment EMAs are unaffected — they remain unbiased
  estimates of E[g] regardless of the noise injected into g.
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
class AdEMAMixState:
    """State for AdEMAMix moment scaling."""

    m_fast: Any
    m_slow: Any
    nu: Any
    phi: Any
    step: int


def _scale_by_ademamix(
    b1: float,
    b2: float,
    b3: float,
    alpha: float,
    eps: float,
    default_noise_stddev: float | PerGroup,
    bc_floor: float,
) -> GradientTransformation:
    _default_per_group = is_per_group(default_noise_stddev)

    def init_fn(params: Any) -> AdEMAMixState:
        zeros = lambda p: torch.zeros_like(p)  # noqa: E731
        phi: Any = (
            {k: 0.0 for k in params} if _default_per_group else 0.0
        )
        return AdEMAMixState(
            m_fast=tree_map(zeros, params),
            m_slow=tree_map(zeros, params),
            nu=tree_map(zeros, params),
            phi=phi,
            step=0,
        )

    def update_fn(
        updates: Any,
        state: AdEMAMixState,
        *,
        params: Any = None,  # noqa: ARG001
        inplace: bool = False,  # noqa: ARG001
        noise_stddev: float | PerGroup | None = None,
        noisy_squared_grads: Any = None,
    ) -> tuple[Any, AdEMAMixState]:
        if noisy_squared_grads is not None and noise_stddev is not None:
            raise ValueError(
                "ademamix.update() received both noisy_squared_grads (JME) and "
                "noise_stddev (DP-BC); pass exactly one (or neither)."
            )

        t = state.step + 1
        new_mf = tree_map(lambda m, g: b1 * m + (1 - b1) * g, state.m_fast, updates)
        new_ms = tree_map(lambda m, g: b3 * m + (1 - b3) * g, state.m_slow, updates)

        bc1 = 1 - b1**t
        bc2 = 1 - b2**t

        # ---- v-update ----------------------------------------------------
        if noisy_squared_grads is not None:
            new_nu = tree_map(
                lambda v, g2: b2 * v + (1 - b2) * g2,
                state.nu,
                noisy_squared_grads,
            )
            new_phi = state.phi
            result = tree_map(
                lambda mf, ms, v: ((mf / bc1) + alpha * ms) / ((v / bc2).sqrt() + eps),
                new_mf,
                new_ms,
                new_nu,
            )
            return result, AdEMAMixState(
                m_fast=new_mf, m_slow=new_ms, nu=new_nu, phi=new_phi, step=t
            )

        new_nu = tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, state.nu, updates)
        effective = (
            noise_stddev if noise_stddev is not None else default_noise_stddev
        )
        per_group = is_per_group(effective) or isinstance(state.phi, dict)

        if per_group:
            assert isinstance(new_mf, dict), (
                "PerGroup BC requires top-level dict params."
            )
            new_phi = {}
            result = {}
            for key in new_mf:
                nv_k = resolve_noise_variance(effective, key)
                old_phi_k = (
                    state.phi[key] if isinstance(state.phi, dict) else state.phi
                )
                new_phi_k = b2 * old_phi_k + (1 - b2) * nv_k
                new_phi[key] = new_phi_k
                phi_hat = new_phi_k / bc2
                if phi_hat > 0:
                    v_hat = torch.clamp(new_nu[key] / bc2 - phi_hat, min=bc_floor)
                else:
                    v_hat = new_nu[key] / bc2
                result[key] = ((new_mf[key] / bc1) + alpha * new_ms[key]) / (
                    v_hat.sqrt() + eps
                )
        else:
            scalar_var = float(effective) ** 2
            new_phi = update_phi_ema(state.phi, scalar_var, b2)
            phi_hat = new_phi / bc2

            if phi_hat > 0:

                def _compute(mf, ms, v):
                    v_hat = torch.clamp(v / bc2 - phi_hat, min=bc_floor)
                    return ((mf / bc1) + alpha * ms) / (v_hat.sqrt() + eps)
            else:

                def _compute(mf, ms, v):
                    return ((mf / bc1) + alpha * ms) / ((v / bc2).sqrt() + eps)

            result = tree_map(_compute, new_mf, new_ms, new_nu)

        return result, AdEMAMixState(
            m_fast=new_mf, m_slow=new_ms, nu=new_nu, phi=new_phi, step=t
        )

    return GradientTransformation(init_fn, update_fn)


def ademamix(
    lr: _LR = 1e-3,
    betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
    alpha: float = 5.0,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    *,
    decoupled_weight_decay: bool = True,
    update_rms_clip: float | None = None,
    noise_stddev: float | PerGroup = 0.0,
) -> GradientTransformation:
    """Create an AdEMAMix optimizer.

    Args:
        lr: Learning rate, scalar or schedule.
        betas: ``(β₁, β₂, β₃)``.  ``β₁`` is the fast first-moment EMA,
            ``β₂`` is the second moment, ``β₃`` is the slow first-moment
            EMA (paper default 0.9999).
        alpha: Weight on the slow EMA in the update mix.
        eps: Denominator stability constant.
        weight_decay: Decoupled WD coefficient by default.
        decoupled_weight_decay: ``False`` switches to L2-style.
        update_rms_clip: Optional StableAdamW-style RMS clip on the
            moment-scaled update.
        noise_stddev: Default DP noise σ for the φ-EMA correction on
            ``v̂``; can be overridden per step at ``update()``.

    Returns:
        A ``torchopt.base.GradientTransformation``.
    """
    if len(betas) != 3:
        raise ValueError(f"betas must contain exactly three values, got {betas}")
    b1, b2, b3 = betas
    for name, b in (("β₁", b1), ("β₂", b2), ("β₃", b3)):
        if not 0 <= b < 1:
            raise ValueError(f"{name} must satisfy 0 <= b < 1, got {b}")
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
    if isinstance(noise_stddev, PerGroup):
        for gname, val in noise_stddev.values.items():
            if val < 0:
                raise ValueError(
                    f"noise_stddev must be non-negative for all groups, "
                    f"got {val} for group '{gname}'"
                )
    elif noise_stddev < 0:
        raise ValueError(f"noise_stddev must be non-negative, got {noise_stddev}")

    bc_floor = eps * eps
    moment = _scale_by_ademamix(
        b1=b1,
        b2=b2,
        b3=b3,
        alpha=alpha,
        eps=eps,
        default_noise_stddev=noise_stddev,
        bc_floor=bc_floor,
    )
    return make_optimizer_chain(
        moment,
        lr=lr,
        weight_decay=weight_decay,
        decoupled_weight_decay=decoupled_weight_decay,
        update_rms_clip=update_rms_clip,
    )


__all__ = ["ademamix", "AdEMAMixState"]
