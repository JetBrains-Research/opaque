"""Shared chain composer for functional optimizers.

Builds a ``torchopt`` ``GradientTransformation`` chain on top of a
moment-scaling primitive (``scale_by_adam`` for AdamW, ``sign-of-momentum``
for Lion, …).  The composer decides where weight decay attaches (decoupled
post-moment vs. L2 pre-moment), whether to clip the update by its RMS
(StableAdamW), and applies the (negative) learning rate at the end.

The composer does **not** know about DP-specific concerns (``noise_stddev``,
``noisy_squared_grads``); those live inside the moment scaler that the caller
passes in.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from typing import Any

try:
    import torchopt
    from torchopt.alias.utils import scale_by_neg_lr
    from torchopt.base import GradientTransformation
except ImportError as exc:
    raise ImportError(
        "torchopt is required for opaque.optimizers. "
        "Install it with: pip install 'torchopt>=0.7.3'"
    ) from exc

import torch

from opaque.core.noise import SecondMomentNoiseOutput
from opaque.core.pytree import tree_map


_LR = float | Callable[[int], float]


def _rms_clip_transform(threshold: float) -> GradientTransformation:
    """Per-update RMS clip used by StableAdamW.

    Computes ``rms = sqrt(mean(update**2))`` over the entire update pytree
    (concatenated leaves) and divides the update by ``max(1, rms / threshold)``.
    No state.
    """

    def init_fn(params: Any) -> tuple:
        del params
        return ()

    def update_fn(
        updates: Any,
        state: tuple,
        *,
        params: Any = None,  # noqa: ARG001
        inplace: bool = False,  # noqa: ARG001
        **kwargs: Any,  # noqa: ARG001
    ) -> tuple[Any, tuple]:
        # Global RMS across all leaves (param-count-weighted mean of squares).
        sq_sum = torch.zeros((), dtype=torch.float64)
        count = 0
        for leaf in _iter_leaves(updates):
            sq_sum = sq_sum + leaf.detach().to(torch.float64).pow(2).sum()
            count += leaf.numel()
        if count == 0:
            return updates, state
        rms = (sq_sum / count).sqrt()
        scale = torch.clamp(rms / threshold, min=1.0).to(torch.float32)
        scale_f = float(scale.item())
        clipped = tree_map(lambda u: u / scale_f, updates)
        return clipped, state

    return GradientTransformation(init_fn, update_fn)


def _iter_leaves(tree: Any):
    """Yield tensor leaves of a (possibly nested) pytree."""
    if isinstance(tree, torch.Tensor):
        yield tree
        return
    if isinstance(tree, dict):
        for v in tree.values():
            yield from _iter_leaves(v)
        return
    if isinstance(tree, (list, tuple)):
        for v in tree:
            yield from _iter_leaves(v)
        return
    # Non-tensor leaves are silently ignored — same convention as
    # ``opaque.core.pytree.tree_leaves``.


def make_optimizer_chain(
    moment_scaler: GradientTransformation,
    lr: _LR,
    weight_decay: float,
    *,
    decoupled_weight_decay: bool = True,
    update_rms_clip: float | None = None,
) -> GradientTransformation:
    """Compose a moment-scaling primitive into a full optimizer.

    Layout (decoupled / AdamW style)::

        moment_scaler -> [rms_clip] -> add_decayed_weights -> scale_by_neg_lr

    Layout (L2 / Adam style)::

        add_decayed_weights -> moment_scaler -> [rms_clip] -> scale_by_neg_lr

    The L2 form folds ``wd * params`` into the gradient *before* moment
    scaling, so weight decay enters the EMAs.  The decoupled form leaves
    moment scaling on raw gradients and applies weight decay to the
    update post-moment-scaling but pre-LR-scaling — this is the standard
    AdamW recipe (Loshchilov & Hutter).

    ``update_rms_clip`` (StableAdamW): when set, divides the update by
    ``max(1, rms / threshold)`` after moment scaling and before WD/LR.
    The clip applies only to the moment-scaled portion of the update,
    not to the weight-decay term.

    The returned ``GradientTransformation`` accepts ``**kwargs`` on
    ``update`` and forwards them to the moment scaler — so DP-aware
    moment scalers can read ``noise_stddev`` / ``noisy_squared_grads``
    from the caller without the chain knowing about them.
    """
    moment_update_params = inspect.signature(moment_scaler.update).parameters
    accepts_second_moment = "noisy_squared_grads" in moment_update_params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in moment_update_params.values()
    )

    def _route_second_moment_output(
        updates: Any,
        kwargs: dict[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        if not isinstance(updates, SecondMomentNoiseOutput):
            return updates, kwargs
        if "noisy_squared_grads" in kwargs:
            raise ValueError(
                "optimizer.update() received SecondMomentNoiseOutput and an explicit "
                "noisy_squared_grads kwarg; pass only one second-moment source."
            )
        if not accepts_second_moment:
            warnings.warn(
                "This optimizer does not consume private second-moment outputs; "
                "using noisy_grads and ignoring noisy_squared_grads.",
                RuntimeWarning,
                stacklevel=3,
            )
            return updates.noisy_grads, kwargs
        routed = dict(kwargs)
        routed["noisy_squared_grads"] = updates.noisy_squared_grads
        return updates.noisy_grads, routed

    wd = torchopt.transform.add_decayed_weights(weight_decay=weight_decay)
    neg_lr = scale_by_neg_lr(lr)
    clip = (
        _rms_clip_transform(float(update_rms_clip))
        if update_rms_clip is not None
        else None
    )

    if decoupled_weight_decay:

        def init_fn(params: Any) -> tuple:
            return (
                moment_scaler.init(params),
                clip.init(params) if clip is not None else (),
                wd.init(params),
                neg_lr.init(params),
            )

        def update_fn(
            updates: Any,
            state: tuple,
            *,
            params: Any = None,
            inplace: bool = False,
            **kwargs: Any,
        ) -> tuple[Any, tuple]:
            s_mom, s_clip, s_wd, s_lr = state
            updates, kwargs = _route_second_moment_output(updates, kwargs)
            updates, s_mom = moment_scaler.update(
                updates, s_mom, params=params, inplace=inplace, **kwargs
            )
            if clip is not None:
                updates, s_clip = clip.update(
                    updates, s_clip, params=params, inplace=inplace
                )
            updates, s_wd = wd.update(updates, s_wd, params=params, inplace=inplace)
            updates, s_lr = neg_lr.update(updates, s_lr, inplace=inplace)
            return updates, (s_mom, s_clip, s_wd, s_lr)

    else:
        # L2 form: wd is added to gradient before the moment scaler.
        def init_fn(params: Any) -> tuple:
            return (
                wd.init(params),
                moment_scaler.init(params),
                clip.init(params) if clip is not None else (),
                neg_lr.init(params),
            )

        def update_fn(  # type: ignore[misc]
            updates: Any,
            state: tuple,
            *,
            params: Any = None,
            inplace: bool = False,
            **kwargs: Any,
        ) -> tuple[Any, tuple]:
            s_wd, s_mom, s_clip, s_lr = state
            updates, kwargs = _route_second_moment_output(updates, kwargs)
            updates, s_wd = wd.update(updates, s_wd, params=params, inplace=inplace)
            updates, s_mom = moment_scaler.update(
                updates, s_mom, params=params, inplace=inplace, **kwargs
            )
            if clip is not None:
                updates, s_clip = clip.update(
                    updates, s_clip, params=params, inplace=inplace
                )
            updates, s_lr = neg_lr.update(updates, s_lr, inplace=inplace)
            return updates, (s_wd, s_mom, s_clip, s_lr)

    return GradientTransformation(init_fn, update_fn)


__all__ = ["make_optimizer_chain"]
