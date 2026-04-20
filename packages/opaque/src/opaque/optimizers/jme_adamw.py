"""JME-AdamW optimizer — convenience wrapper around :func:`dp_adamw`.

Thin wrapper that creates a :func:`dp_adamw` instance with
``weight_decay=0`` (matching the JME paper's Adam formulation).

The ``noisy_squared_grads`` kwarg on ``update()`` is forwarded to
``dp_adamw``'s unified moment scaler.  See :func:`dp_adamw` for full
documentation.

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

from collections.abc import Callable

try:
    from torchopt.base import GradientTransformation
except ImportError as exc:
    raise ImportError(
        "torchopt is required for opaque.optimizers. "
        "Install it with: pip install 'torchopt>=0.7.3'"
    ) from exc

from opaque.optimizers.dp_adamw import DPAdamWState

# JmeAdamWState is now DPAdamWState (superset — has phi field).
JmeAdamWState = DPAdamWState


def jme_adamw(
    lr: float | Callable[[int], float] = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> GradientTransformation:
    """Create a JME-AdamW optimizer (AdamW with JME dual-stream noise).

    Convenience wrapper around :func:`dp_adamw` with ``weight_decay=0``
    by default (matching the JME paper).  Pass ``noisy_squared_grads``
    to ``update()`` to use JME's privately-estimated g^2.

    With ``weight_decay=0`` (default) this is plain Adam per the JME
    paper.  With ``weight_decay > 0`` it is AdamW.

    Args:
        lr: Learning rate — a float or a callable ``step -> float``.
        betas: ``(beta1, beta2)`` for moment estimation.
        eps: Denominator stability constant.
        weight_decay: Decoupled weight decay (default 0 = pure Adam).

    Returns:
        A ``torchopt.base.GradientTransformation``.

    References:
        - Kalinin, Upadhyay, Lampert (2025) "Continual Release Moment
          Estimation with Differential Privacy" https://arxiv.org/abs/2502.06597
    """
    from opaque.optimizers.dp_adamw import dp_adamw

    return dp_adamw(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


__all__ = ["jme_adamw", "JmeAdamWState"]
