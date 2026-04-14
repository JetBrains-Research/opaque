"""JME-AdamW optimizer — Adam with JME dual-stream noise.

Paired with :func:`~opaque.noise.mf.jme_noise`, this optimizer
consumes noisy gradients (first moment) and noisy squared gradients
(second moment) produced by the JME mechanism (arXiv:2502.06597).

Follows ``dp_adamw``'s AdamW composition (decoupled weight decay)::

    chain(
        scale_by_jme_adamw,                 # custom: external second moment
        add_decayed_weights,               # reused from torchopt (decoupled)
        scale_by_neg_lr,                   # reused from torchopt
    )

With ``weight_decay=0`` (default) this is plain Adam per the JME paper.

Usage::

    from opaque.optimizers import jme_adamw

    optimizer = jme_adamw(lr=1e-3, betas=(0.9, 0.999))
    opt_state = optimizer.init(params)

    (noisy_grads, noisy_sq), noise_state = noise_fn(clipped_grads, noise_state)
    updates, opt_state = optimizer.update(
        noisy_grads, opt_state, noisy_squared_grads=noisy_sq,
    )
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

from opaque.utils.pytree import tree_map


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class JmeAdamWState:
    """State for the JME moment-scaling transform.

    First element of the chain state returned by :func:`jme_adamw`.

    Attributes:
        mu: First-moment EMA (pytree matching params).
        nu: Second-moment EMA.
        step: Update count (1-indexed after first call).
    """

    mu: Any
    nu: Any
    step: int


# ---------------------------------------------------------------------------
# Internal: moment scaling with external second moment
# ---------------------------------------------------------------------------


def _scale_by_jme_adamw(
    b1: float,
    b2: float,
    eps: float,
) -> GradientTransformation:
    """Adam moment scaling with externally-provided second moments.

    Identical to ``torchopt.transform.scale_by_adam`` except the second
    moment uses ``noisy_squared_grads`` from JME instead of squaring the
    gradient input (``order=2``).
    """

    def init_fn(params: Any) -> JmeAdamWState:
        return JmeAdamWState(
            mu=tree_map(torch.zeros_like, params),
            nu=tree_map(torch.zeros_like, params),
            step=0,
        )

    def update_fn(
        updates: Any,
        state: JmeAdamWState,
        *,
        params: Any = None,
        inplace: bool = False,
        noisy_squared_grads: Any = None,
    ) -> tuple[Any, JmeAdamWState]:
        if noisy_squared_grads is None:
            raise ValueError(
                "jme_adamw requires noisy_squared_grads from jme_noise(). "
                "Pass noisy_squared_grads to optimizer.update()."
            )

        t = state.step + 1

        new_mu = tree_map(lambda m, g: b1 * m + (1 - b1) * g, state.mu, updates)
        new_nu = tree_map(
            lambda v, g2: b2 * v + (1 - b2) * g2, state.nu, noisy_squared_grads,
        )

        bc1 = 1 - b1**t
        bc2 = 1 - b2**t

        result = tree_map(
            lambda m, v: (m / bc1) / ((v / bc2).sqrt() + eps),
            new_mu, new_nu,
        )

        new_state = JmeAdamWState(mu=new_mu, nu=new_nu, step=t)
        return result, new_state

    return GradientTransformation(init_fn, update_fn)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def jme_adamw(
    lr: float | Callable[[int], float] = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> GradientTransformation:
    """Create a JME-AdamW optimizer (AdamW with JME dual-stream noise).

    Composes (same order as ``dp_adamw`` from PR #119):

    1. JME moment scaling (custom — external ``noisy_squared_grads``)
    2. Decoupled weight decay (``torchopt.transform.add_decayed_weights``)
    3. Learning rate (``scale_by_neg_lr`` — supports callable schedules)

    With ``weight_decay=0`` (default) this is plain Adam per the JME
    paper.  With ``weight_decay > 0`` it is AdamW (decoupled — weight
    decay bypasses the moment EMAs).

    The ``noisy_squared_grads`` kwarg on ``update()`` provides the
    privately-estimated second moment from :func:`~opaque.noise.mf.jme_noise`.

    Args:
        lr: Learning rate — a float or a callable ``step -> float``
            for LR schedules.
        betas: ``(beta1, beta2)`` for moment estimation.
        eps: Denominator stability constant.
        weight_decay: Decoupled weight decay (default 0 = pure Adam).

    Returns:
        A ``torchopt.base.GradientTransformation``.

    References:
        - Kalinin, Upadhyay, Lampert (2025) "Continual Release Moment
          Estimation with Differential Privacy" https://arxiv.org/abs/2502.06597

    Example::

        optimizer = jme_adamw(lr=1e-3, weight_decay=0.01)
        opt_state = optimizer.init(params)

        (noisy_grads, noisy_sq), noise_state = noise_fn(grads, noise_state)
        updates, opt_state = optimizer.update(
            noisy_grads, opt_state,
            params=params, noisy_squared_grads=noisy_sq,
        )
        params = torchopt.apply_updates(params, updates)
    """
    import torchopt
    from torchopt.alias.utils import scale_by_neg_lr

    b1, b2 = betas

    adam_scale = _scale_by_jme_adamw(b1, b2, eps)
    wd = torchopt.transform.add_decayed_weights(weight_decay=weight_decay)
    neg_lr = scale_by_neg_lr(lr)

    def init_fn(params: Any) -> tuple:
        return (adam_scale.init(params), wd.init(params), neg_lr.init(params))

    def update_fn(
        updates: Any,
        state: tuple,
        *,
        params: Any = None,
        inplace: bool = False,
        noisy_squared_grads: Any = None,
    ) -> tuple[Any, tuple]:
        s_adam, s_wd, s_lr = state

        updates, s_adam = adam_scale.update(
            updates, s_adam,
            params=params, inplace=inplace,
            noisy_squared_grads=noisy_squared_grads,
        )
        updates, s_wd = wd.update(updates, s_wd, params=params, inplace=inplace)
        updates, s_lr = neg_lr.update(updates, s_lr, inplace=inplace)

        return updates, (s_adam, s_wd, s_lr)

    return GradientTransformation(init_fn, update_fn)


__all__ = ["jme_adamw", "JmeAdamWState"]
