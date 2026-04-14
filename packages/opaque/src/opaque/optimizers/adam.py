"""DP-Adam optimizer for use with JME noise.

Returns a ``(init, update)`` pair matching the ``torchopt``
:class:`~torchopt.base.GradientTransformation` protocol, so it
plugs directly into training loops that already use ``torchopt.sgd``.

The key difference from standard Adam: the caller provides **noisy
squared gradients** (from JME's second noise stream) alongside the
noisy gradients.  These are passed via the ``noisy_squared_grads``
keyword argument to ``update``.

Usage::

    from opaque.optimizers import dp_adam

    optimizer = dp_adam(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)
    opt_state = optimizer.init(params)

    # In training loop:
    noisy_grads, noise_state = noise_fn(clipped_grads, noise_state)
    updates, opt_state = optimizer.update(
        noisy_grads, opt_state,
        noisy_squared_grads=noise_state.noisy_squared_grads,
    )
    params = torchopt.apply_updates(params, updates)
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, NamedTuple

import torch

from opaque.utils.pytree import tree_map


@dataclasses.dataclass(frozen=True)
class DPAdamState:
    """Optimizer state for :func:`dp_adam`.

    Attributes:
        m: First-moment EMA (same pytree shape as params).
        v: Second-moment EMA.
        step: Number of ``update`` calls made (1-indexed after first call).
    """

    m: Any
    v: Any
    step: int


class DPAdamTransformation(NamedTuple):
    """``torchopt``-compatible ``(init, update)`` pair for DP-Adam."""

    init: Callable[[Any], DPAdamState]
    update: Callable[..., tuple[Any, DPAdamState]]


def dp_adam(
    lr: float | Callable[[int], float] = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> DPAdamTransformation:
    """Create a DP-Adam optimizer.

    Returns an ``(init, update)`` named tuple matching the ``torchopt``
    :class:`~torchopt.base.GradientTransformation` protocol.

    Unlike ``torchopt.adam``, this optimizer does **not** compute the
    second moment from the gradients.  Instead, the caller supplies
    pre-computed *noisy squared gradients* (from
    :func:`~opaque.noise.mf.mf_noise_jme`) via the
    ``noisy_squared_grads`` keyword argument to ``update``.

    Args:
        lr: Learning rate — a float or a callable ``step -> float``.
        beta1: First-moment decay (default 0.9).
        beta2: Second-moment decay (default 0.999).
        eps: Denominator epsilon (default 1e-8).

    Returns:
        A ``DPAdamTransformation(init, update)`` named tuple.

    Example::

        optimizer = dp_adam(lr=1e-3)
        opt_state = optimizer.init(params)

        noisy_grads, noise_state = noise_fn(grads, noise_state)
        updates, opt_state = optimizer.update(
            noisy_grads, opt_state,
            noisy_squared_grads=noise_state.noisy_squared_grads,
        )
        params = torchopt.apply_updates(params, updates)
    """

    def _lr(step: int) -> float:
        return lr(step) if callable(lr) else lr

    def init(params: Any) -> DPAdamState:
        return DPAdamState(
            m=tree_map(torch.zeros_like, params),
            v=tree_map(torch.zeros_like, params),
            step=0,
        )

    def update(
        updates: Any,
        state: DPAdamState,
        *,
        noisy_squared_grads: Any,
        params: Any | None = None,  # unused, kept for torchopt compat
        inplace: bool = False,  # unused, kept for torchopt compat
    ) -> tuple[Any, DPAdamState]:
        """Compute Adam parameter updates.

        Args:
            updates: Noisy gradients (first moment input).
            state: Current optimizer state.
            noisy_squared_grads: Noisy element-wise squared gradients
                from ``noise_state.noisy_squared_grads``.
            params: Unused (present for ``torchopt`` compatibility).
            inplace: Unused (present for ``torchopt`` compatibility).

        Returns:
            ``(updates, new_state)`` where ``updates`` is the pytree
            of parameter deltas (add to params).
        """
        new_step = state.step + 1

        new_m = tree_map(
            lambda m, g: beta1 * m + (1.0 - beta1) * g,
            state.m, updates,
        )
        new_v = tree_map(
            lambda v, g2: beta2 * v + (1.0 - beta2) * g2,
            state.v, noisy_squared_grads,
        )

        bc1 = 1.0 - beta1 ** new_step
        bc2 = 1.0 - beta2 ** new_step
        current_lr = _lr(new_step - 1)

        param_updates = tree_map(
            lambda m, v: -current_lr * (m / bc1) / (torch.sqrt(v / bc2) + eps),
            new_m, new_v,
        )

        new_state = DPAdamState(m=new_m, v=new_v, step=new_step)
        return param_updates, new_state

    return DPAdamTransformation(init=init, update=update)
