"""DP-AdamW optimizer: AdamW with optional DP bias correction.

Implements Algorithm 1 (DP-AdamW) and Algorithm 2 (DP-AdamW-BC) from::

    Chooi et al., "DP-AdamW: Investigating Decoupled Weight Decay and Bias
    Correction in Private Deep Learning", arXiv:2511.07843 (ICML 2025).

Algorithm 1 (``noise_variance=0``, default):
    Standard AdamW applied to (already noised) gradients.  Moment scaling
    is mathematically identical to ``torchopt.transform.scale_by_adam``;
    weight decay and learning-rate application reuse torchopt transforms.

Algorithm 2 (``noise_variance > 0``):
    DP-AdamW-BC.  Tracks an exponential moving average of the per-step
    noise variance using the same beta_2 coefficient as the second moment,
    then subtracts the bias-corrected noise EMA from v-hat_t::

        phi_t = beta_2 * phi_{t-1} + (1-beta_2) * Phi_t
        v_hat_corrected = max(v_hat_t - phi_t/(1-beta_2^t), gamma)

    This correctly handles adaptive clipping (where Phi_t varies per step)
    and reduces to the paper's constant subtraction when Phi is fixed.

Both variants follow TorchOpt's ``GradientTransformation`` protocol::

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
    import torchopt
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
    """Adam moment scaling with optional DP bias correction.

    Equivalent to ``torchopt.transform.scale_by_adam`` when
    ``noise_variance=0`` and no per-step override is given.

    When noise variance is active, tracks a beta_2-EMA of the per-step
    noise variance and subtracts it (bias-corrected) from v-hat::

        phi_t = beta_2 * phi_{t-1} + (1-beta_2) * Phi_t
        v_hat_corrected = max(v_hat_t - phi_t/(1-beta_2^t), bc_floor)

    The ``noise_variance`` kwarg on ``update_fn`` overrides the default
    for that step (e.g. when adaptive clipping changes the sensitivity).
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
    ) -> tuple[Any, DPAdamWState]:
        effective_nv = noise_variance if noise_variance is not None else default_nv
        t = state.step + 1

        # Moment updates (paper steps 3-4).
        new_mu = tree_map(lambda m, g: b1 * m + (1 - b1) * g, state.mu, updates)
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

    When ``noise_variance=0`` (default), the moment scaling is identical to
    ``torchopt.transform.scale_by_adam`` (Algorithm 1).

    When ``noise_variance > 0``, the second moment is bias-corrected by
    tracking a beta_2-EMA of the noise variance and subtracting the
    bias-corrected EMA from v-hat (Algorithm 2, DP-AdamW-BC)::

        phi_t = beta_2 * phi_{t-1} + (1-beta_2) * Phi_t
        v_hat_corrected = max(v_hat - phi_t/(1-beta_2^t), bc_floor)

    The ``noise_variance`` parameter sets the default.  It can be
    overridden per step via ``opt.update(..., noise_variance=current_phi)``
    — the same pattern as :func:`~opaque.noise.gaussian_noise` where the
    constructor sets a default stddev and each call can override it.  This
    is necessary with adaptive clipping, where the sensitivity (and thus
    the injected noise variance) changes every step.

    When ``noise_variance`` is a :class:`~opaque.utils.per_group.PerGroup`
    (from MSE-optimal per-group noise allocation), each parameter group
    has its own noise **stddev** looked up and squared internally.

    Weight decay and learning-rate application reuse
    ``torchopt.transform.add_decayed_weights`` and
    ``torchopt.transform.scale``.

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
            (as returned by
            :func:`~opaque.noise.per_group_noise_stddev`) applies
            per-group correction (each group's stddev is squared
            internally).
            When ``0``, moment scaling matches standard AdamW exactly.
            Can be overridden per step in ``.update()``.
        bc_floor: Minimum value gamma for the corrected second moment.
            Prevents division by zero in the BC variant.

    Returns:
        A ``torchopt.base.GradientTransformation`` with ``.init`` and
        ``.update`` methods.  The ``.update`` method accepts an optional
        ``noise_variance`` keyword to override the default for that step.

    Example (Algorithm 1 -- standard DP-AdamW)::

        >>> opt = dp_adamw(lr=1e-4, weight_decay=0.01)
        >>> state = opt.init(params)
        >>> updates, state = opt.update(noisy_grads, state, params=params)
        >>> params = torchopt.apply_updates(params, updates)

    Example (Algorithm 2 -- DP-AdamW-BC, fixed noise)::

        >>> noise_stddev = noise_multiplier * clip_state.sensitivity
        >>> opt = dp_adamw(lr=1e-4, noise_variance=noise_stddev ** 2)
        >>> state = opt.init(params)
        >>> updates, state = opt.update(noisy_grads, state, params=params)

    Example (Algorithm 2 -- DP-AdamW-BC, adaptive clipping)::

        >>> opt = dp_adamw(lr=1e-4, noise_variance=initial_nv)
        >>> state = opt.init(params)
        >>> # Each step, pass current noise variance:
        >>> current_nv = (noise_multiplier * clip_state.sensitivity) ** 2
        >>> updates, state = opt.update(grads, state, params=params,
        ...                             noise_variance=current_nv)

    Example (Algorithm 2 -- DP-AdamW-BC, per-group noise)::

        >>> from opaque.noise import per_group_noise_stddev
        >>> stddev = per_group_noise_stddev(clip_state, noise_multiplier)
        >>> opt = dp_adamw(lr=1e-4, noise_variance=stddev)

    References:
        Chooi et al., "DP-AdamW: Investigating Decoupled Weight Decay and
        Bias Correction in Private Deep Learning", arXiv:2511.07843 (2025).
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
    # Manual composition (not torchopt.chain) so that the noise_variance kwarg
    # reaches _scale_by_adam_bc's update_fn.
    # Uses scale_by_neg_lr (not scale(-lr)) to support callable LR schedules.
    from torchopt.alias.utils import scale_by_neg_lr

    adam_bc = _scale_by_adam_bc(betas[0], betas[1], eps, noise_variance, bc_floor)
    wd = torchopt.transform.add_decayed_weights(weight_decay=weight_decay)
    neg_lr = scale_by_neg_lr(lr)

    def init_fn(params: Any) -> tuple:
        return (adam_bc.init(params), wd.init(params), neg_lr.init(params))

    def update_fn(
        updates: Any,
        state: tuple,
        *,
        params: Any = None,
        inplace: bool = False,
        noise_variance: float | PerGroup | None = None,
    ) -> tuple[Any, tuple]:
        s_adam, s_wd, s_lr = state
        updates, s_adam = adam_bc.update(
            updates,
            s_adam,
            params=params,
            inplace=inplace,
            noise_variance=noise_variance,
        )
        updates, s_wd = wd.update(updates, s_wd, params=params, inplace=inplace)
        updates, s_lr = neg_lr.update(updates, s_lr, inplace=inplace)
        return updates, (s_adam, s_wd, s_lr)

    return GradientTransformation(init_fn, update_fn)


__all__ = ["dp_adamw", "DPAdamWState"]
