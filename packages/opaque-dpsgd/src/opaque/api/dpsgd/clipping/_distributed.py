"""Distributed synchronization helpers for DP-SGD-specific clipping.

Handles the adaptive clipping state (:class:`AdaptiveClipState`) and
its auxiliary output (:class:`AdaptiveClippedGradAux`). Registers the
sync handlers at import time so that :func:`opaque.distributed.sync`
dispatches correctly once this module is loaded (which happens via
``import opaque.dpsgd`` or via constructing an :class:`AdaptiveClipState`).
"""

from __future__ import annotations

from dataclasses import replace

import torch

    assert_scalar_equal,
from opaque.api.engine.clipping._distributed import sync_clipped_grad_aux
from opaque.api.engine.distributed._state import (
    reduce_scalar,
    register_sync_type,
    sync_object,
)
from opaque.distributed import is_distributed
from opaque.random import fold_in
from opaque.torch.random import generator_from_key
from opaque.types import PerGroup

from ._adaptive import (
    AdaptiveClippedGradAux,
    AdaptiveClipState,
    _adaptive_clipping_norm_update,
    _sample_noisy_clipping_rate,
)

_ADAPTIVE_CLIP_STATE_FIELD_OPS: dict[str, str] = {
    "_current_clipping_norm": "local",
    "_next_clipping_norm": "local",
    "_step": "local",
    "_rng_key": "local",
    "_fraction_noise_std": "local",
    "_learning_rate": "local",
    "_target_quantile": "local",
    "_clipping_norm_min": "local",
    "_clipping_norm_max": "local",
    "_num_clipped": "sum",
    "_batch_size": "sum",
}

__all__ = [
    "sync_adaptive_clip_state",
    "sync_adaptive_clipped_grad_aux",
]
def _assert_rng_key_equal(state: AdaptiveClipState) -> None:
    """Assert that all ranks derive adaptive threshold noise from one key."""
    assert_scalar_equal(int(state._rng_key.seed), name="AdaptiveClipState.seed")




def sync_adaptive_clip_state(state: AdaptiveClipState) -> AdaptiveClipState:
    """Recompute adaptive clipping state from globally aggregated local counts."""
    if not is_distributed():
    _assert_rng_key_equal(state)

        return state

    is_per_group = isinstance(state._num_clipped, dict)

    if is_per_group:
        global_batch_size = reduce_scalar(state._batch_size, op="sum")

        global_num_clipped: dict[str, float] = {}
        for gname, local_count in state._num_clipped.items():
            global_num_clipped[gname] = reduce_scalar(local_count, op="sum")

        if global_batch_size == 0:
            return replace(state, _batch_size=global_batch_size)

        current_pg = state._current_clipping_norm
        step_for_noise = max(0, state._step - 1)
        new_values: dict[str, float] = {}
        for i, gname in enumerate(sorted(current_pg.values.keys())):
            global_rate = global_num_clipped[gname] / max(1.0, global_batch_size)
            group_key = fold_in(fold_in(state._rng_key, step_for_noise), i)
            generator = generator_from_key(group_key)
            noise = (
                torch.randn(1, generator=generator).item() * state._fraction_noise_std
            )
            noisy_rate = global_rate + noise

            new_values[gname] = _adaptive_clipping_norm_update(
                base_clipping_norm=current_pg.values[gname],
                noisy_clipping_rate=noisy_rate,
                target_quantile=state._target_quantile,
                learning_rate=state._learning_rate,
                clipping_norm_min=state._clipping_norm_min,
                clipping_norm_max=state._clipping_norm_max,
            )

        new_clipping_norm = PerGroup(groups=current_pg.groups, values=new_values)
        return replace(
            state,
            _next_clipping_norm=new_clipping_norm,
            _num_clipped=global_num_clipped,
            _batch_size=global_batch_size,
        )

    synced = sync_object(
        state,
        field_ops=_ADAPTIVE_CLIP_STATE_FIELD_OPS,
    )

    if synced._batch_size == 0:
        return synced

    global_rate = synced._num_clipped / max(1.0, synced._batch_size)

    step_for_noise = max(0, synced._step - 1)
    noisy_global_rate = _sample_noisy_clipping_rate(
        global_rate,
        key=synced._rng_key,
        step=step_for_noise,
        fraction_noise_std=synced._fraction_noise_std,
    )

    new_clipping_norm = _adaptive_clipping_norm_update(
        base_clipping_norm=synced._current_clipping_norm,
        noisy_clipping_rate=noisy_global_rate,
        target_quantile=synced._target_quantile,
        learning_rate=synced._learning_rate,
        clipping_norm_min=synced._clipping_norm_min,
        clipping_norm_max=synced._clipping_norm_max,
    )

    return replace(
        synced,
        _next_clipping_norm=float(new_clipping_norm),
    )


def sync_adaptive_clipped_grad_aux(
    aux: AdaptiveClippedGradAux,
) -> AdaptiveClippedGradAux:
    """Synchronize ``AdaptiveClippedGradAux`` across distributed ranks.

    Delegates to :func:`opaque.api.engine.clipping._distributed.sync_clipped_grad_aux`
    which handles ``ClippedGradAux`` subclasses generically.
    """
    if not is_distributed():
        return aux
    return sync_clipped_grad_aux(aux)


register_sync_type(AdaptiveClipState, sync_adaptive_clip_state)
register_sync_type(AdaptiveClippedGradAux, sync_adaptive_clipped_grad_aux)
