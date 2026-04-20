"""Shared AdamW composition: moment_scaler → weight_decay → neg_lr."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

try:
    import torchopt
    from torchopt.base import GradientTransformation
except ImportError as exc:
    raise ImportError(
        "torchopt is required for opaque.optimizers. "
        "Install it with: pip install 'torchopt>=0.7.3'"
    ) from exc


def _adamw_chain(
    moment_scaler: GradientTransformation,
    lr: float | Callable[[int], float],
    weight_decay: float,
) -> GradientTransformation:
    """Compose ``moment_scaler → add_decayed_weights → scale_by_neg_lr``.

    Both :func:`~opaque.optimizers.adamw_bc` and
    :func:`~opaque.optimizers.adamw_jme` share this identical three-stage
    AdamW composition.  The only difference is the moment scaler (which
    may accept custom kwargs like ``noise_stddev`` or
    ``noisy_squared_grads``).  Extra ``**kwargs`` on ``update()`` are
    forwarded to the moment scaler.
    """
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
