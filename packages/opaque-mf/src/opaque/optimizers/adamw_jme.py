"""AdamW-JME optimizer: AdamW with JME dual-stream noise.

Paired with :func:`~opaque.noise.mf.jme_noise`, this optimizer
consumes noisy gradients (first moment) and noisy squared gradients
(second moment) produced by the JME mechanism::

    Kalinin, Upadhyay, Lampert, "Continual Release Moment Estimation
    with Differential Privacy", arXiv:2502.06597 (2025).

Instead of squaring the noised gradient (which amplifies noise in
the second moment), JME provides a separately privatized estimate
of g² via a matrix-factorization mechanism.

Follows TorchOpt's ``GradientTransformation`` protocol::

    state = opt.init(params)
    updates, state = opt.update(grads, state, params=params,
                                noisy_squared_grads=noisy_sq)
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

from opaque.core.utils.pytree import tree_map


# ---------------------------------------------------------------------------
# AdamW chain: moment_scaler -> add_decayed_weights -> scale_by_neg_lr
# ---------------------------------------------------------------------------


def _adamw_chain(
    moment_scaler: GradientTransformation,
    lr: float | Callable[[int], float],
    weight_decay: float,
) -> GradientTransformation:
    """Compose ``moment_scaler -> add_decayed_weights -> scale_by_neg_lr``."""
    from torchopt.alias.utils import scale_by_neg_lr

    wd = torchopt.transform.add_decayed_weights(weight_decay=weight_decay)
    neg_lr = scale_by_neg_lr(lr)

    def init_fn(params: Any) -> tuple:
        return (moment_scaler.init(params), wd.init(params), neg_lr.init(params))

    def update_fn(
        updates: Any,
        state: tuple,
        *,
        params: Any = None,
        inplace: bool = False,
        **kwargs: Any,
    ) -> tuple[Any, tuple]:
        s_adam, s_wd, s_lr = state
        updates, s_adam = moment_scaler.update(
            updates, s_adam, params=params, inplace=inplace, **kwargs
        )
        updates, s_wd = wd.update(updates, s_wd, params=params, inplace=inplace)
        updates, s_lr = neg_lr.update(updates, s_lr, inplace=inplace)
        return updates, (s_adam, s_wd, s_lr)

    return GradientTransformation(init_fn, update_fn)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AdamWJMEState:
    """Immutable state for the JME moment-scaling transform.

    First element of the chain state returned by :func:`adamw_jme`.

    Attributes:
        mu: First-moment EMA (pytree matching params).
        nu: Second-moment EMA (pytree matching params).
        step: Update count (1-indexed after first call).
    """

    mu: Any
    nu: Any
    step: int


# ---------------------------------------------------------------------------
# Internal: moment scaling with external second moment
# ---------------------------------------------------------------------------


def _scale_by_adam_jme(
    b1: float,
    b2: float,
    eps: float,
) -> GradientTransformation:
    """Adam moment scaling with externally-provided second moments.

    Identical to ``torchopt.transform.scale_by_adam`` except the second
    moment uses ``noisy_squared_grads`` from JME instead of squaring the
    gradient input.
    """

    def init_fn(params: Any) -> AdamWJMEState:
        return AdamWJMEState(
            mu=tree_map(torch.zeros_like, params),
            nu=tree_map(torch.zeros_like, params),
            step=0,
        )

    def update_fn(
        updates: Any,
        state: AdamWJMEState,
        *,
        params: Any = None,
        inplace: bool = False,
        noisy_squared_grads: Any = None,
    ) -> tuple[Any, AdamWJMEState]:
        if noisy_squared_grads is None:
            raise ValueError(
                "adamw_jme requires noisy_squared_grads from jme_noise(). "
                "Pass noisy_squared_grads to optimizer.update()."
            )

        t = state.step + 1

        new_mu = tree_map(lambda m, g: b1 * m + (1 - b1) * g, state.mu, updates)
        new_nu = tree_map(
            lambda v, g2: b2 * v + (1 - b2) * g2,
            state.nu,
            noisy_squared_grads,
        )

        bc1 = 1 - b1**t
        bc2 = 1 - b2**t

        result = tree_map(
            lambda m, v: (m / bc1) / ((v / bc2).sqrt() + eps),
            new_mu,
            new_nu,
        )

        new_state = AdamWJMEState(mu=new_mu, nu=new_nu, step=t)
        return result, new_state

    return GradientTransformation(init_fn, update_fn)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def adamw_jme(
    lr: float | Callable[[int], float] = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> GradientTransformation:
    """Create an AdamW-JME optimizer (AdamW with JME dual-stream noise).

    Composes:

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

    Example::

        optimizer = adamw_jme(lr=1e-3, weight_decay=0.01)
        opt_state = optimizer.init(params)

        (noisy_grads, noisy_sq), noise_state = noise_fn(grads, noise_state)
        updates, opt_state = optimizer.update(
            noisy_grads, opt_state,
            params=params, noisy_squared_grads=noisy_sq,
        )
        params = torchopt.apply_updates(params, updates)

    References:
        Kalinin, Upadhyay, Lampert, "Continual Release Moment Estimation
        with Differential Privacy", arXiv:2502.06597 (2025).
    """
    b1, b2 = betas
    adam_scale = _scale_by_adam_jme(b1, b2, eps)
    return _adamw_chain(adam_scale, lr, weight_decay)


__all__ = ["adamw_jme", "AdamWJMEState"]
