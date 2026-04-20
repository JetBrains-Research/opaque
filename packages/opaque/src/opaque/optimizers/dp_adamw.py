"""DP-AdamW optimizer with optional bias correction and external second moments.

Unified AdamW for DP training.  Three features compose independently:

1. **Standard AdamW** (defaults) — identical to ``torchopt.adamw``.
2. **Bias correction** (``noise_variance > 0``) — Algorithm 2 from
   Chooi et al. (arXiv:2511.07843, ICML 2025).  Subtracts the known
   DP noise variance from the second-moment EMA.
3. **External second moments** (``noisy_squared_grads`` at update time) —
   uses privately-estimated g² from the JME mechanism (Kalinin et al.,
   arXiv:2502.06597) instead of squaring the noised gradient.

Features 2 and 3 are orthogonal: BC corrects for *known* noise in v,
while JME provides a *better* v estimate.  They can be combined.

Follows TorchOpt's ``GradientTransformation`` protocol::

    state = opt.init(params)
    updates, state = opt.update(grads, state, params=params)
    params = torchopt.apply_updates(params, updates)
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

from opaque.utils.per_group import PerGroup
from opaque.utils.pytree import tree_map


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DPAdamWState:
    """Immutable state for the bias-corrected moment scaling (Algorithm 2).

    This is the first element of the chain state returned by
    :func:`dp_adamw`.  The full optimizer state is a tuple
    ``(DPAdamWState, wd_state, lr_state)``.

    Attributes:
        mu: First moment estimates (pytree matching params).
        nu: Second moment estimates (pytree matching params).
        phi: Noise variance EMA.  A ``float`` for scalar noise_variance,
            or ``dict[str, float]`` for per-group noise_variance.  Tracks
            the beta_2-weighted running average of the per-step noise
            variance Phi_t injected into gradients.
        step: Number of update steps completed.
    """

    mu: Any
    nu: Any
    phi: Any
    step: int


# ---------------------------------------------------------------------------
# Internal: bias-corrected moment scaling (replaces torchopt.scale_by_adam)
# ---------------------------------------------------------------------------


def _resolve_nv_scalar(nv: float | PerGroup, key: str) -> float:
    """Resolve noise variance to a scalar for a given param key.

    Scalar input is returned as-is (already variance = stddev**2).
    PerGroup input contains stddevs; returns stddev**2 for the key's group.
    """
    if isinstance(nv, PerGroup):
        return nv.for_key(key) ** 2
    return nv


def _scale_by_adam_bc(
    b1: float,
    b2: float,
    eps: float,
    default_nv: float | PerGroup,
    bc_floor: float,
) -> GradientTransformation:
    """Adam moment scaling with optional BC and external second moments.

    Equivalent to ``torchopt.transform.scale_by_adam`` when
    ``noise_variance=0`` and ``noisy_squared_grads`` is not provided.

    Optional features (compose independently):

    * **noise_variance** — tracks a beta_2-EMA of the per-step noise
      variance and subtracts it (bias-corrected) from v-hat.
    * **noisy_squared_grads** — uses externally-provided g² (e.g. from
      JME) for the second moment instead of squaring the gradient.
    """
    _default_per_group = isinstance(default_nv, PerGroup)

    def init_fn(params: Any) -> DPAdamWState:
        mu = tree_map(torch.zeros_like, params)
        nu = tree_map(torch.zeros_like, params)
        phi: float | dict[str, float]
        if _default_per_group:
            phi = {k: 0.0 for k in params}
        else:
            phi = 0.0
        return DPAdamWState(mu=mu, nu=nu, phi=phi, step=0)

    def update_fn(
        updates: Any,
        state: DPAdamWState,
        *,
        params: Any = None,
        inplace: bool = False,
        noise_variance: float | PerGroup | None = None,
        noisy_squared_grads: Any = None,
    ) -> tuple[Any, DPAdamWState]:
        effective_nv = noise_variance if noise_variance is not None else default_nv
        t = state.step + 1

        # First moment: always EMA of the gradient.
        new_mu = tree_map(lambda m, g: b1 * m + (1 - b1) * g, state.mu, updates)

        # Second moment: use external g² (JME) when provided, else self-square.
        if noisy_squared_grads is not None:
            new_nu = tree_map(
                lambda v, g2: b2 * v + (1 - b2) * g2,
                state.nu,
                noisy_squared_grads,
            )
        else:
            new_nu = tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, state.nu, updates)

        # Bias correction denominators.
        bc1 = 1 - b1**t
        bc2 = 1 - b2**t

        per_group = isinstance(effective_nv, PerGroup) or isinstance(state.phi, dict)

        if per_group:
            # Per-group path: each parameter has its own noise variance EMA.
            result = {}
            new_phi: dict[str, float] = {}
            for key in new_mu:
                nv_k = _resolve_nv_scalar(effective_nv, key)
                old_phi_k = state.phi[key] if isinstance(state.phi, dict) else state.phi
                new_phi_k = b2 * old_phi_k + (1 - b2) * nv_k
                new_phi[key] = new_phi_k

                m_hat = new_mu[key] / bc1
                phi_hat = new_phi_k / bc2
                if phi_hat > 0:
                    v_hat = torch.clamp(new_nu[key] / bc2 - phi_hat, min=bc_floor)
                else:
                    v_hat = new_nu[key] / bc2
                result[key] = m_hat / (v_hat.sqrt() + eps)
        else:
            # Scalar path: same noise variance EMA for all parameters.
            scalar_nv = float(effective_nv)
            new_phi_scalar = b2 * state.phi + (1 - b2) * scalar_nv
            phi_hat = new_phi_scalar / bc2

            if phi_hat > 0:

                def _compute(m: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                    m_hat = m / bc1
                    v_hat = torch.clamp(v / bc2 - phi_hat, min=bc_floor)
                    return m_hat / (v_hat.sqrt() + eps)
            else:

                def _compute(m: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                    return (m / bc1) / ((v / bc2).sqrt() + eps)

            result = tree_map(_compute, new_mu, new_nu)
            new_phi = new_phi_scalar

        new_state = DPAdamWState(mu=new_mu, nu=new_nu, phi=new_phi, step=t)
        return result, new_state

    return GradientTransformation(init_fn, update_fn)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dp_adamw(
    lr: float | Callable[[int], float] = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    *,
    noise_variance: float | PerGroup = 0.0,
    bc_floor: float = 1e-8,
) -> GradientTransformation:
    """Create a DP-AdamW optimizer.

    With default arguments this is standard AdamW, identical to
    ``torchopt.adamw``.  Two optional features compose independently:

    **Bias correction** (``noise_variance > 0``):
        Tracks a beta_2-EMA of the noise variance and subtracts
        it from v-hat (Algorithm 2, DP-AdamW-BC, arXiv:2511.07843)::

            phi_t = beta_2 * phi_{t-1} + (1-beta_2) * Phi_t
            v_hat_corrected = max(v_hat - phi_t/(1-beta_2^t), bc_floor)

    **External second moments** (``noisy_squared_grads`` at update time):
        Uses privately-estimated g^2 from JME (arXiv:2502.06597) instead
        of squaring the noised gradient.  Reduces noise amplification in
        the second moment.

    The optimizer expects gradients that are already clipped and noised.
    Clipping and noise injection are separate concerns handled by
    :func:`opaque.clipped_grad` and :func:`opaque.gaussian_noise`.

    Args:
        lr: Learning rate eta — a float or a callable ``step -> float``
            for LR schedules.
        betas: Coefficients (beta_1, beta_2) for moment estimation.
        eps: Denominator stability constant epsilon.
        weight_decay: Decoupled weight decay coefficient lambda.
        noise_variance: Default DP noise variance Phi for bias correction.
            A scalar ``stddev**2`` applies uniformly.
            A :class:`~opaque.utils.per_group.PerGroup` of **stddevs**
            applies per-group correction.
            When ``0`` (default), no bias correction is applied.
            Can be overridden per step in ``.update()``.
        bc_floor: Minimum value gamma for the corrected second moment.
            Prevents division by zero in the BC variant.

    Returns:
        A ``torchopt.base.GradientTransformation``.  The ``.update``
        method accepts optional kwargs:

        * ``noise_variance`` — override the default for that step.
        * ``noisy_squared_grads`` — pytree of privately-estimated g^2
          (from :func:`~opaque.noise.mf.jme_noise`).

    Example (standard AdamW on noised gradients)::

        >>> opt = dp_adamw(lr=1e-4, weight_decay=0.01)
        >>> state = opt.init(params)
        >>> updates, state = opt.update(noisy_grads, state, params=params)

    Example (bias correction, fixed noise)::

        >>> noise_stddev = noise_multiplier * clip_state.sensitivity
        >>> opt = dp_adamw(lr=1e-4, noise_variance=noise_stddev ** 2)

    Example (JME external second moments)::

        >>> opt = dp_adamw(lr=1e-3, weight_decay=0.0)
        >>> (noisy_grads, noisy_sq), noise_state = jme_noise_fn(grads, noise_state)
        >>> updates, state = opt.update(
        ...     noisy_grads, state, noisy_squared_grads=noisy_sq,
        ... )

    References:
        - Chooi et al., "DP-AdamW: Investigating Decoupled Weight Decay
          and Bias Correction", arXiv:2511.07843 (ICML 2025).
        - Kalinin, Upadhyay, Lampert, "Continual Release Moment Estimation
          with Differential Privacy", arXiv:2502.06597 (2025).
    """
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if bc_floor <= 0:
        raise ValueError(f"bc_floor must be positive, got {bc_floor}")
    if len(betas) != 2:
        raise ValueError(f"betas must contain exactly two values, got {betas}")
    b1, b2 = betas
    if not 0 <= b1 < 1:
        raise ValueError(f"beta_1 must satisfy 0 <= beta_1 < 1, got {b1}")
    if not 0 <= b2 < 1:
        raise ValueError(f"beta_2 must satisfy 0 <= beta_2 < 1, got {b2}")
    if isinstance(noise_variance, PerGroup):
        for gname, val in noise_variance.values.items():
            if val < 0:
                raise ValueError(
                    f"noise_variance stddev must be non-negative for all groups, "
                    f"got {val} for group '{gname}'"
                )
    elif noise_variance < 0:
        raise ValueError(f"noise_variance must be non-negative, got {noise_variance}")

    # Compose: moment scaling (custom) + weight decay + lr.
    # Uses _adamw_chain which forwards **kwargs (noise_variance) to the
    # moment scaler's update_fn.
    from opaque.optimizers._chain import _adamw_chain

    adam_bc = _scale_by_adam_bc(betas[0], betas[1], eps, noise_variance, bc_floor)
    return _adamw_chain(adam_bc, lr, weight_decay)


__all__ = ["dp_adamw", "DPAdamWState"]
