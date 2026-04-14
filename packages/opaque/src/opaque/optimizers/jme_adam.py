"""JME-Adam optimizer — Adam with JME dual-stream noise.

Paired with :func:`~opaque.noise.mf.jme_noise`, this optimizer
consumes noisy gradients (first moment) and noisy squared gradients
(second moment) produced by the JME mechanism (arXiv:2502.06597).

Follows the ``torchopt.GradientTransformation`` protocol and reuses
``torchopt.transform.add_decayed_weights`` and ``torchopt.transform.scale``
for weight decay and learning-rate application (same pattern as
:func:`~opaque.optimizers.dp_adamw`).

Usage::

    from opaque.optimizers import jme_adam

    optimizer = jme_adam(lr=1e-3, beta1=0.9, beta2=0.999)
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
class JmeAdamState:
    """State for the JME moment-scaling transform.

    First element of the chain state returned by :func:`jme_adam`.
    The full optimizer state is ``(JmeAdamState, wd_state, lr_state)``.

    Attributes:
        mu: First-moment EMA (pytree matching params).
        nu: Second-moment EMA.
        step: Update count (1-indexed after first call).
    """

    mu: Any
    nu: Any
    step: int


# ---------------------------------------------------------------------------
# Internal: moment scaling (replaces torchopt.transform.scale_by_adam)
# ---------------------------------------------------------------------------


def _scale_by_jme_adam(
    b1: float,
    b2: float,
    eps: float,
    lr: float | Callable[[int], float],
) -> GradientTransformation:
    """Adam moment scaling with externally-provided second moments.

    Like ``torchopt.transform.scale_by_adam``, but the second moment
    uses noisy squared gradients from JME instead of squaring the
    (already-noisy) gradient input.  Includes LR application (supports
    callables for LR schedules).
    """

    def _get_lr(step: int) -> float:
        return lr(step) if callable(lr) else lr

    def init_fn(params: Any) -> JmeAdamState:
        return JmeAdamState(
            mu=tree_map(torch.zeros_like, params),
            nu=tree_map(torch.zeros_like, params),
            step=0,
        )

    def update_fn(
        updates: Any,
        state: JmeAdamState,
        *,
        params: Any = None,
        inplace: bool = False,
        noisy_squared_grads: Any = None,
    ) -> tuple[Any, JmeAdamState]:
        if noisy_squared_grads is None:
            raise ValueError(
                "jme_adam requires noisy_squared_grads from jme_noise(). "
                "Pass noisy_squared_grads to optimizer.update()."
            )

        t = state.step + 1

        new_mu = tree_map(lambda m, g: b1 * m + (1 - b1) * g, state.mu, updates)
        new_nu = tree_map(
            lambda v, g2: b2 * v + (1 - b2) * g2, state.nu, noisy_squared_grads,
        )

        bc1 = 1 - b1**t
        bc2 = 1 - b2**t
        current_lr = _get_lr(state.step)

        result = tree_map(
            lambda m, v: -current_lr * (m / bc1) / ((v / bc2).sqrt() + eps),
            new_mu, new_nu,
        )

        new_state = JmeAdamState(mu=new_mu, nu=new_nu, step=t)
        return result, new_state

    return GradientTransformation(init_fn, update_fn)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def jme_adam(
    lr: float | Callable[[int], float] = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> GradientTransformation:
    """Create a JME-Adam optimizer (Adam with JME dual-stream noise).

    Follows the ``torchopt.GradientTransformation`` protocol.  Composes:

    1. JME moment scaling + LR (custom — uses external ``noisy_squared_grads``)
    2. Weight decay (``torchopt.transform.add_decayed_weights``)

    The ``noisy_squared_grads`` kwarg on ``update()`` provides the
    privately-estimated second moment from :func:`~opaque.noise.mf.jme_noise`.

    Args:
        lr: Learning rate — a float or a callable ``step -> float``
            for LR schedules.
        betas: ``(beta1, beta2)`` for moment estimation.
        eps: Denominator stability constant.
        weight_decay: Decoupled weight decay (default 0, set >0 for AdamW).

    Returns:
        A ``torchopt.base.GradientTransformation``.

    References:
        - Kalinin, Upadhyay, Lampert (2025) "Continual Release Moment
          Estimation with Differential Privacy" https://arxiv.org/abs/2502.06597

    Example::

        optimizer = jme_adam(lr=1e-3, weight_decay=0.01)
        opt_state = optimizer.init(params)

        (noisy_grads, noisy_sq), noise_state = noise_fn(grads, noise_state)
        updates, opt_state = optimizer.update(
            noisy_grads, opt_state, noisy_squared_grads=noisy_sq,
        )
        params = torchopt.apply_updates(params, updates)
    """
    adam = _scale_by_jme_adam(betas[0], betas[1], eps, lr)
    wd = torchopt.transform.add_decayed_weights(weight_decay=weight_decay)

    def init_fn(params: Any) -> tuple:
        return (adam.init(params), wd.init(params))

    def update_fn(
        updates: Any,
        state: tuple,
        *,
        params: Any = None,
        inplace: bool = False,
        noisy_squared_grads: Any = None,
    ) -> tuple[Any, tuple]:
        s_adam, s_wd = state
        updates, s_adam = adam.update(
            updates, s_adam,
            params=params, inplace=inplace,
            noisy_squared_grads=noisy_squared_grads,
        )
        updates, s_wd = wd.update(updates, s_wd, params=params, inplace=inplace)
        return updates, (s_adam, s_wd)

    return GradientTransformation(init_fn, update_fn)


__all__ = ["jme_adam", "JmeAdamState"]
