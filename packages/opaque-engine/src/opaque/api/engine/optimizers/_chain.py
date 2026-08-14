"""Shared chain composer for backend-neutral optimizers.

Builds a ``(step_fn, state)`` optimizer on top of a moment-scaling
primitive (``_scale_by_adam`` for AdamW, ``_sgd_step`` for SGD).  The
composer decides where weight decay attaches (decoupled post-moment vs.
L2 pre-moment), whether to clip the update by its RMS (StableAdamW), and
applies the (negative) learning rate at the end.

The composer understands the public clipped/noised metadata wrappers and
routes their DP metadata into the moment scaler when the scaler accepts
it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from opaque.api.engine import ops
from opaque.pytree import tree_map
from opaque.types import ClippedPytree, NoisedPytree, SecondMomentNoiseOutput

_LR = float | Callable[[int], float]
_MomentStep = Callable[
    [Any, Any, Any, float | None, Any | None],
    tuple[Any, Any],
]


def apply_updates(params: Any, updates: Any) -> Any:
    """Apply signed updates to a parameter pytree.

    Pure helper: ``new_params = params + updates`` leafwise.
    """
    return tree_map(lambda p, u: ops.add(p, u), params, updates)


def _clip_by_global_rms(updates: Any, threshold: float) -> Any:
    """Apply one parameter-count-weighted RMS scale to all tensor leaves."""
    leaves = _tensor_leaves(updates)
    if not leaves:
        return updates

    total = None
    count = 0
    for leaf in leaves:
        acc_dtype = ops.accumulator_dtype(leaf, kind="sum")
        leaf_cast = ops.astype(leaf, acc_dtype)
        sq = ops.square(leaf_cast)
        leaf_sum = ops.sum(sq, dtype=acc_dtype)
        total = leaf_sum if total is None else ops.add(total, leaf_sum)
        count += _numel(leaf)

    rms = ops.sqrt(ops.divide(total, count))
    scale = ops.clamp(ops.divide(rms, threshold), lo=1.0)

    return tree_map(
        lambda u: ops.divide(u, ops.astype(scale, ops.dtype(u))),
        updates,
    )


def _tensor_leaves(tree: Any) -> list[Any]:
    """Yield tensor leaves of a (possibly nested) pytree."""
    if ops.is_array(tree):
        return [tree]
    if isinstance(tree, dict):
        leaves: list[Any] = []
        for v in tree.values():
            leaves.extend(_tensor_leaves(v))
        return leaves
    if isinstance(tree, (list, tuple)):
        leaves = []
        for v in tree:
            leaves.extend(_tensor_leaves(v))
        return leaves
    return []


def _numel(leaf: Any) -> int:
    shape = ops.shape(leaf)
    total = 1
    for dim in shape:
        total *= dim
    return total


def _route_noisy_pytree(updates: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(updates, NoisedPytree):
        return updates.pytree, {"noise_stddev": updates.noise_stddev}
    if isinstance(updates, ClippedPytree):
        raise TypeError(
            "optimizer step received ClippedPytree updates that have not "
            "passed through a noise mechanism. Pass NoisedPytree outputs from "
            "a DP mechanism, or unwrap `.pytree` explicitly for non-private use."
        )
    return updates, {}


def _unwrap_second_moment_value(value: Any, *, name: str) -> Any:
    if isinstance(value, NoisedPytree):
        return value.pytree
    if isinstance(value, ClippedPytree):
        raise TypeError(
            f"SecondMomentNoiseOutput.{name} is a ClippedPytree that has not "
            "passed through a noise mechanism."
        )
    return value


def _route_second_moment_output(updates: Any) -> tuple[Any, dict[str, Any]]:
    if not isinstance(updates, SecondMomentNoiseOutput):
        return _route_noisy_pytree(updates)
    grads, routed = _route_noisy_pytree(updates.noisy_grads)
    return (
        grads,
        {
            **routed,
            "noisy_squared_grads": _unwrap_second_moment_value(
                updates.noisy_squared_grads, name="noisy_squared_grads"
            ),
        },
    )


def make_optimizer_chain(
    moment_step: _MomentStep,
    moment_init_state: Any,
    lr: _LR,
    weight_decay: float,
    *,
    decoupled_weight_decay: bool = True,
    update_rms_clip: float | None = None,
    maximize: bool = False,
) -> tuple[Callable[[Any, Any, Any], tuple[Any, Any]], Any]:
    """Compose a moment-scaling primitive into a full optimizer.

    Layout (decoupled / AdamW style)::

        moment_scaler -> [rms_clip] -> add_decayed_weights -> scale_by_neg_lr

    Layout (L2 / Adam / SGD style)::

        [maximize sign flip] -> add_decayed_weights -> moment_scaler
        -> [rms_clip] -> scale_by_neg_lr

    ``update_rms_clip`` (StableAdamW): when set, divides the update by
    ``max(1, rms / threshold)`` after moment scaling and before WD/LR.

    The returned ``step_fn`` extracts DP metadata from ``NoisedPytree`` /
    ``SecondMomentNoiseOutput`` updates and threads it into the moment
    scaler internally — there is no public per-step metadata kwarg.
    """

    def step_fn(
        updates: Any,
        state: Any,
        *,
        params: Any,
    ) -> tuple[Any, Any]:
        current_step = state.step
        lr_value = lr(current_step) if callable(lr) else lr

        raw, routed = _route_second_moment_output(updates)

        if maximize:
            raw = tree_map(lambda g: ops.multiply(g, -1.0), raw)

        if decoupled_weight_decay:
            scaled, new_state = moment_step(
                raw,
                state,
                params,
                routed.get("noise_stddev"),
                routed.get("noisy_squared_grads"),
            )
            if update_rms_clip is not None:
                scaled = _clip_by_global_rms(scaled, update_rms_clip)
            if weight_decay != 0.0:
                scaled = tree_map(
                    lambda u, p: ops.add(u, ops.multiply(p, weight_decay)),
                    scaled,
                    params,
                )
        else:
            if weight_decay != 0.0:
                raw = tree_map(
                    lambda g, p: ops.add(g, ops.multiply(p, weight_decay)),
                    raw,
                    params,
                )
            scaled, new_state = moment_step(
                raw,
                state,
                params,
                routed.get("noise_stddev"),
                routed.get("noisy_squared_grads"),
            )
            if update_rms_clip is not None:
                scaled = _clip_by_global_rms(scaled, update_rms_clip)

        if lr_value != 0.0:
            scaled = tree_map(lambda u: ops.multiply(u, -lr_value), scaled)

        return scaled, new_state

    return step_fn, moment_init_state


__all__ = ["apply_updates", "make_optimizer_chain"]
