"""Universal Adam / AdamW factory.

Single entry point for the Adam family.  Handles four orthogonal
behaviors selected at factory time and at update time:

1. **L2 vs decoupled weight decay** — constructor flag
   ``decoupled_weight_decay``.  ``True`` is AdamW (Loshchilov &
   Hutter); ``False`` is the original Adam with L2 regularisation
   folded into the gradient.

2. **StableAdamW RMS clip** — constructor knob ``update_rms_clip``.
   ``None`` disables the clip; a positive float divides the moment-scaled
   update by ``max(1, rms / threshold)``.  See Wortsman et al.,
   "Stable and low-precision training for large-scale vision-language
   models" (2023).

3. **DP noise-variance bias correction** — pass ``noise_stddev`` either
   at construction (default) or at every ``update()`` call (per-step
   override; useful under adaptive clipping where σ varies).  When
   non-zero, the second moment is corrected by a β₂-EMA of the noise
   variance — Chooi et al., "DP-AdamW", arXiv:2511.07843.

4. **Private second-moment stream** — pass ``noisy_squared_grads``
   to ``update()`` to bypass squaring the noised gradient and use a
   privately-estimated ``g²`` instead.  Kalinin, Upadhyay, Lampert,
   "Continual Release Moment Estimation with Differential Privacy",
   arXiv:2502.06597.

Modes (3) and (4) are mutually exclusive at any single ``update()``
call: passing both raises ``ValueError`` (the optimizer would have
to discard one or the other).  Either may be active when the other
is not.

The optimizer follows torchopt's ``GradientTransformation`` protocol::

    opt = adamw(lr=1e-3, weight_decay=0.01)
    state = opt.init(params)

    # Vanilla AdamW:
    updates, state = opt.update(grads, state, params=p)

    # DP-AdamW-BC (per-step σ override):
    updates, state = opt.update(noisy_grads, state, params=p,
                                noise_stddev=current_sigma)

    # DP-AdamW with a private second-moment stream:
    updates, state = opt.update(noisy_grads, state, params=p,
                                noisy_squared_grads=noisy_sq)
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
    init_per_group_phi,
    is_per_group,
    resolve_noise_variance,
    update_phi_ema,
)
from opaque.optimizers._chain import make_optimizer_chain


_LR = float | Callable[[int], float]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AdamState:
    """Immutable state for Adam-family moment scaling.

    Carries the noise-variance EMA ``phi`` regardless of whether DP
    bias correction is actively in use — this keeps the state shape
    constant across calls so checkpoints don't depend on call history.
    The cost is one scalar (or one ``dict[str, float]`` per group).

    Attributes:
        mu: First-moment EMA (pytree matching params).
        nu: Second-moment EMA (pytree matching params).
        phi: Noise-variance EMA (scalar or ``dict[group, float]``).
            Stays at zero unless ``noise_stddev`` is passed at update
            time.
        step: Number of completed updates.
    """

    mu: Any
    nu: Any
    phi: Any
    step: int


# ---------------------------------------------------------------------------
# Moment scaler — handles vanilla / BC / external second-moment branches
# ---------------------------------------------------------------------------


def _scale_by_adam(
    b1: float,
    b2: float,
    eps: float,
    default_noise_stddev: float | PerGroup,
    bc_floor: float,
) -> GradientTransformation:
    """Adam moment scaling with optional DP bias correction or private second moments.

    Update modes selected by the kwargs passed to ``update()``:

    - ``noisy_squared_grads`` not None: v-update consumes the externally
      privatised second-moment stream directly.  ``noise_stddev`` is
        ignored for this step (the second-moment post-processing argument means no
      φ-EMA correction is needed).

    - ``noise_stddev`` non-zero (or default non-zero): v-update squares
      the noised gradient, then subtracts the bias-corrected φ-EMA::

          v̂_corrected = max(v̂ − φ̂, floor)

    - both absent (and default ``noise_stddev==0``): standard Adam.
    """
    _default_per_group = is_per_group(default_noise_stddev)

    def init_fn(params: Any) -> AdamState:
        mu = tree_map(torch.zeros_like, params)
        nu = tree_map(torch.zeros_like, params)
        phi: Any = init_per_group_phi(params) if _default_per_group else 0.0
        return AdamState(mu=mu, nu=nu, phi=phi, step=0)

    def update_fn(
        updates: Any,
        state: AdamState,
        *,
        params: Any = None,  # noqa: ARG001
        inplace: bool = False,  # noqa: ARG001
        noise_stddev: float | PerGroup | None = None,
        noisy_squared_grads: Any = None,
    ) -> tuple[Any, AdamState]:
        if noisy_squared_grads is not None and noise_stddev is not None:
            raise ValueError(
                "adamw.update() received both noisy_squared_grads and "
                "noise_stddev (DP-BC); these select mutually exclusive v-update "
                "branches.  Pass exactly one (or neither, for vanilla AdamW)."
            )

        t = state.step + 1

        # First moment is the same in all branches.
        new_mu = tree_map(lambda m, g: b1 * m + (1 - b1) * g, state.mu, updates)

        # ---- v-update ----------------------------------------------------
        if noisy_squared_grads is not None:
            # External second-moment branch: g² stream replaces (g·g).  No φ-EMA
            # correction (post-processing already gave us an unbiased v).
            new_nu = tree_map(
                lambda v, g2: b2 * v + (1 - b2) * g2,
                state.nu,
                noisy_squared_grads,
            )
            new_phi = state.phi  # unchanged
            bc1 = 1 - b1**t
            bc2 = 1 - b2**t
            result = tree_map(
                lambda m, v: (
                    (m / bc1) / (torch.clamp(v / bc2, min=bc_floor).sqrt() + eps)
                ),
                new_mu,
                new_nu,
            )
            return result, AdamState(mu=new_mu, nu=new_nu, phi=new_phi, step=t)

        # Standard / BC branch: square the (possibly noised) gradient.
        new_nu = tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, state.nu, updates)

        effective_stddev = (
            noise_stddev if noise_stddev is not None else default_noise_stddev
        )
        per_group = is_per_group(effective_stddev) or isinstance(state.phi, dict)

        bc1 = 1 - b1**t
        bc2 = 1 - b2**t

        if per_group:
            # Per-leaf path: walk ``new_mu`` and ``new_nu`` in lockstep
            # by dotted-key paths matching :class:`PerGroup`'s lookup
            # keys, so nested param pytrees work the same as flat dicts.
            new_phi: dict[str, float] = {}

            def _bc_walk(mu_node: Any, nu_node: Any, prefix: str) -> Any:
                if isinstance(mu_node, dict):
                    return {
                        k: _bc_walk(
                            mu_node[k],
                            nu_node[k],
                            f"{prefix}.{k}" if prefix else str(k),
                        )
                        for k in mu_node
                    }
                # Tensor leaf.
                path = prefix
                nv = resolve_noise_variance(effective_stddev, path)
                old_phi_k = (
                    state.phi.get(path, 0.0)
                    if isinstance(state.phi, dict)
                    else state.phi
                )
                new_phi_k = b2 * old_phi_k + (1 - b2) * nv
                new_phi[path] = new_phi_k
                m_hat = mu_node / bc1
                phi_hat = new_phi_k / bc2
                if phi_hat > 0:
                    v_hat = torch.clamp(nu_node / bc2 - phi_hat, min=bc_floor)
                else:
                    v_hat = nu_node / bc2
                return m_hat / (v_hat.sqrt() + eps)

            result = _bc_walk(new_mu, new_nu, "")
        else:
            scalar_var = float(effective_stddev) ** 2
            new_phi = update_phi_ema(state.phi, scalar_var, b2)
            phi_hat = new_phi / bc2

            if phi_hat > 0:

                def _compute(m: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                    return (m / bc1) / (
                        torch.clamp(v / bc2 - phi_hat, min=bc_floor).sqrt() + eps
                    )
            else:

                def _compute(m: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                    return (m / bc1) / ((v / bc2).sqrt() + eps)

            result = tree_map(_compute, new_mu, new_nu)

        return result, AdamState(mu=new_mu, nu=new_nu, phi=new_phi, step=t)

    return GradientTransformation(init_fn, update_fn)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def adamw(
    lr: _LR = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    *,
    decoupled_weight_decay: bool = True,
    update_rms_clip: float | None = None,
    noise_stddev: float | PerGroup = 0.0,
) -> GradientTransformation:
    """Universal Adam / AdamW factory.

    Args:
        lr: Learning rate, scalar or ``step → float`` callable schedule.
        betas: ``(β₁, β₂)`` coefficients for first / second moment EMAs.
        eps: Denominator stability constant.
        weight_decay: Weight-decay coefficient (decoupled by default).
        decoupled_weight_decay: ``True`` selects AdamW (decoupled WD,
            modern default).  ``False`` selects the original Adam, where
            ``weight_decay * params`` is added to the gradient before
            moment scaling — i.e. L2 regularisation enters the EMAs.
        update_rms_clip: When not ``None``, divides the moment-scaled
            update by ``max(1, rms / threshold)`` (StableAdamW).  ``rms``
            is the global root-mean-square over all tensor leaves.
        noise_stddev: Default per-step noise σ (scalar or
            :class:`PerGroup`) used for DP-AdamW-BC bias correction.
            ``0.0`` (default) disables the correction; a non-zero value
            activates it.  Can be overridden per step at ``update()``.

    Returns:
        A ``torchopt.base.GradientTransformation``.

    DP usage notes:

    - At ``update()`` time, pass ``noise_stddev=current_σ`` to override
      the constructor default (necessary under adaptive clipping where
      σ changes per step).
    - Alternatively pass ``noisy_squared_grads`` to consume an externally
        privatised second-moment stream; ``noise_stddev`` is ignored
      for that step.
    - Passing both ``noise_stddev`` and ``noisy_squared_grads`` at the
      same call raises ``ValueError``.
    """
    _validate(eps, betas, weight_decay, update_rms_clip, noise_stddev)
    bc_floor = eps * eps  # see module docstring on the rationale.
    moment = _scale_by_adam(
        b1=betas[0],
        b2=betas[1],
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


def _validate(
    eps: float,
    betas: tuple[float, float],
    weight_decay: float,
    update_rms_clip: float | None,
    noise_stddev: float | PerGroup,
) -> None:
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if len(betas) != 2:
        raise ValueError(f"betas must contain exactly two values, got {betas}")
    b1, b2 = betas
    if not 0 <= b1 < 1:
        raise ValueError(f"beta_1 must satisfy 0 <= beta_1 < 1, got {b1}")
    if not 0 <= b2 < 1:
        raise ValueError(f"beta_2 must satisfy 0 <= beta_2 < 1, got {b2}")
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


__all__ = ["adamw", "AdamState"]
