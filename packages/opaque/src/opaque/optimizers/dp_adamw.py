"""DP-AdamW optimizer: AdamW with optional DP bias correction.

Implements Algorithm 1 (DP-AdamW) and Algorithm 2 (DP-AdamW-BC) from::

    Chooi et al., "DP-AdamW: Investigating Decoupled Weight Decay and Bias
    Correction in Private Deep Learning", arXiv:2511.07843 (ICML 2025).

Algorithm 1 (``noise_variance=0``, default):
    Standard AdamW applied to (already noised) gradients.  Moment scaling
    is mathematically identical to ``torchopt.transform.scale_by_adam``;
    weight decay and learning-rate application reuse torchopt transforms.

Algorithm 2 (``noise_variance > 0``):
    DP-AdamW-BC.  Subtracts the DP noise variance |Phi| from the bias-corrected
    second moment estimate v-hat_t, clamped to a floor gamma, to undo the
    upward bias that Gaussian noise injection introduces.

Both variants follow TorchOpt's ``GradientTransformation`` protocol::

    state = opt.init(params)
    updates, state = opt.update(grads, state, params=params)
    params = torchopt.apply_updates(params, updates)
"""

from __future__ import annotations

import dataclasses
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

from opaque.utils.pytree import tree_map


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DPAdamWState:
    """Immutable state for the bias-corrected moment scaling (Algorithm 2).

    This is the first element of the chain state returned by
    :func:`dp_adamw`.  The full optimizer state is a tuple
    ``(DPAdamWState, wd_state, lr_state)`` managed by ``torchopt.chain``.

    Attributes:
        mu: First moment estimates (pytree matching params).
        nu: Second moment estimates (pytree matching params).
        step: Number of update steps completed.
    """

    mu: Any
    nu: Any
    step: int


# ---------------------------------------------------------------------------
# Internal: bias-corrected moment scaling (replaces torchopt.scale_by_adam)
# ---------------------------------------------------------------------------


def _scale_by_adam_bc(
    b1: float,
    b2: float,
    eps: float,
    noise_variance: float,
    bc_floor: float,
) -> GradientTransformation:
    """Adam moment scaling with optional DP bias correction.

    Equivalent to ``torchopt.transform.scale_by_adam`` when
    ``noise_variance=0``.  When ``noise_variance > 0``, subtracts the DP
    noise variance from the bias-corrected second moment (Algorithm 2)::

        v_hat_corrected = max(v_hat - noise_variance, bc_floor)
        output = m_hat / (sqrt(v_hat_corrected) + eps)
    """

    def init_fn(params: Any) -> DPAdamWState:
        mu = tree_map(torch.zeros_like, params)
        nu = tree_map(torch.zeros_like, params)
        return DPAdamWState(mu=mu, nu=nu, step=0)

    def update_fn(
        updates: Any,
        state: DPAdamWState,
        *,
        params: Any = None,
        inplace: bool = False,
    ) -> tuple[Any, DPAdamWState]:
        t = state.step + 1

        # Moment updates (paper steps 3-4).
        new_mu = tree_map(lambda m, g: b1 * m + (1 - b1) * g, state.mu, updates)
        new_nu = tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, state.nu, updates)

        # Bias correction (paper steps 5-6) + BC noise subtraction.
        bc1 = 1 - b1**t
        bc2 = 1 - b2**t

        def _compute(m: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            m_hat = m / bc1
            v_hat = v / bc2
            if noise_variance > 0:
                v_hat = torch.clamp(v_hat - noise_variance, min=bc_floor)
            return m_hat / (v_hat.sqrt() + eps)

        result = tree_map(_compute, new_mu, new_nu)
        new_state = DPAdamWState(mu=new_mu, nu=new_nu, step=t)
        return result, new_state

    return GradientTransformation(init_fn, update_fn)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dp_adamw(
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    *,
    noise_variance: float = 0.0,
    bc_floor: float = 1e-8,
) -> GradientTransformation:
    """Create a DP-AdamW optimizer.

    When ``noise_variance=0`` (default), the moment scaling is identical to
    ``torchopt.transform.scale_by_adam`` (Algorithm 1).

    When ``noise_variance > 0``, the second moment is bias-corrected by
    subtracting the DP noise variance (Algorithm 2, DP-AdamW-BC)::

        v_hat_corrected = max(v_hat - noise_variance, bc_floor)

    In both cases, weight decay and learning-rate application reuse
    ``torchopt.transform.add_decayed_weights`` and
    ``torchopt.transform.scale`` via ``torchopt.chain``.

    The optimizer expects gradients that are already clipped and noised.
    Clipping and noise injection are separate concerns handled by
    :func:`opaque.clipped_grad` and :func:`opaque.gaussian_noise`.

    Args:
        lr: Learning rate eta.
        betas: Coefficients (beta_1, beta_2) for moment estimation.
        eps: Denominator stability constant epsilon.
        weight_decay: Decoupled weight decay coefficient lambda.
        noise_variance: DP noise variance Phi = stddev**2 for bias correction.
            When 0, moment scaling matches standard AdamW exactly.
        bc_floor: Minimum value gamma for the corrected second moment.
            Prevents division by zero in the BC variant.

    Returns:
        A ``torchopt.base.GradientTransformation`` with ``.init`` and
        ``.update`` methods.

    Example (Algorithm 1 -- standard DP-AdamW)::

        >>> opt = dp_adamw(lr=1e-4, weight_decay=0.01)
        >>> state = opt.init(params)
        >>> updates, state = opt.update(noisy_grads, state, params=params)
        >>> params = torchopt.apply_updates(params, updates)

    Example (Algorithm 2 -- DP-AdamW-BC)::

        >>> noise_stddev = noise_multiplier * clip_state.sensitivity
        >>> opt = dp_adamw(lr=1e-4, noise_variance=noise_stddev ** 2)
        >>> state = opt.init(params)
        >>> updates, state = opt.update(noisy_grads, state, params=params)
        >>> params = torchopt.apply_updates(params, updates)

    References:
        Chooi et al., "DP-AdamW: Investigating Decoupled Weight Decay and
        Bias Correction in Private Deep Learning", arXiv:2511.07843 (2025).
    """
    if noise_variance < 0:
        raise ValueError(f"noise_variance must be non-negative, got {noise_variance}")

    # Chain: moment scaling (custom) + weight decay + lr (both from torchopt).
    # When noise_variance=0, _scale_by_adam_bc is identical to scale_by_adam.
    return torchopt.chain(
        _scale_by_adam_bc(betas[0], betas[1], eps, noise_variance, bc_floor),
        torchopt.transform.add_decayed_weights(weight_decay=weight_decay),
        torchopt.transform.scale(-lr),
    )


__all__ = ["dp_adamw", "DPAdamWState"]
