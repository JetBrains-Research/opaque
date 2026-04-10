"""DP-AdamW optimizer: AdamW with optional DP bias correction.

Implements Algorithm 1 (DP-AdamW) and Algorithm 2 (DP-AdamW-BC) from::

    Chooi et al., "DP-AdamW: Investigating Decoupled Weight Decay and Bias
    Correction in Private Deep Learning", arXiv:2511.07843 (ICML 2025).

Algorithm 1 (``noise_variance=0``, default):
    Standard AdamW applied to (already noised) gradients.  Delegates entirely
    to ``torchopt.adamw`` -- no custom math.

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


@dataclasses.dataclass(frozen=True)
class DPAdamWState:
    """Immutable state for DP-AdamW-BC optimizer (Algorithm 2).

    When ``noise_variance=0``, :func:`dp_adamw` delegates to
    ``torchopt.adamw`` and uses TorchOpt's own state type instead.

    Attributes:
        mu: First moment estimates (pytree matching params).
        nu: Second moment estimates (pytree matching params).
        step: Number of update steps completed.
    """

    mu: Any
    nu: Any
    step: int


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

    When ``noise_variance=0`` (default), returns ``torchopt.adamw`` directly --
    standard AdamW with no modifications (Algorithm 1).

    When ``noise_variance > 0``, returns DP-AdamW-BC (Algorithm 2) which
    subtracts the DP noise variance from the second moment estimate::

        v_hat_corrected = max(v_hat - noise_variance, bc_floor)

    The optimizer expects gradients that are already clipped and noised.
    Clipping and noise injection are separate concerns handled by
    :func:`opaque.clipped_grad` and :func:`opaque.gaussian_noise`.

    Args:
        lr: Learning rate eta.
        betas: Coefficients (beta_1, beta_2) for moment estimation.
        eps: Denominator stability constant epsilon.
        weight_decay: Decoupled weight decay coefficient lambda.
        noise_variance: DP noise variance Phi = stddev**2 for bias correction.
            When 0, disables BC and delegates to ``torchopt.adamw``.
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

    # Algorithm 1: standard AdamW -- delegate entirely to torchopt.
    if noise_variance == 0:
        return torchopt.adamw(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    # Algorithm 2: DP-AdamW-BC with bias-corrected second moment.
    b1, b2 = betas

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
        if weight_decay != 0 and params is None:
            raise ValueError(
                "params must be passed to update() when weight_decay != 0 "
                "(use opt.update(grads, state, params=params))"
            )

        t = state.step + 1

        # Moment updates (paper steps 3-4).
        new_mu = tree_map(lambda m, g: b1 * m + (1 - b1) * g, state.mu, updates)
        new_nu = tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, state.nu, updates)

        # Bias-correction denominators (paper steps 5-6).
        bc1 = 1 - b1**t
        bc2 = 1 - b2**t

        # DP-AdamW-BC update (paper step 7, Algorithm 2):
        #   v_hat_corrected = max(v_hat - Phi, gamma)
        #   update = -lr * (m_hat / (sqrt(v_hat_corrected) + eps) + wd * theta)
        if params is not None and weight_decay != 0:

            def _step(m: torch.Tensor, v: torch.Tensor, p: torch.Tensor):
                m_hat = m / bc1
                v_hat = torch.clamp(v / bc2 - noise_variance, min=bc_floor)
                return -(lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * p))

            result = tree_map(_step, new_mu, new_nu, params)
        else:

            def _step_no_wd(m: torch.Tensor, v: torch.Tensor):
                m_hat = m / bc1
                v_hat = torch.clamp(v / bc2 - noise_variance, min=bc_floor)
                return -(lr * m_hat / (v_hat.sqrt() + eps))

            result = tree_map(_step_no_wd, new_mu, new_nu)

        new_state = DPAdamWState(mu=new_mu, nu=new_nu, step=t)
        return result, new_state

    return GradientTransformation(init_fn, update_fn)


__all__ = ["dp_adamw", "DPAdamWState"]
