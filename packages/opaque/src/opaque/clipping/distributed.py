"""Distributed synchronization helpers for clipping components.

This module contains clipping-specific distributed logic. Core collectives and
generic utilities remain in ``opaque.distributed``.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from opaque.distributed import (
    assert_scalar_equal,
    gather_pytree,
    is_distributed,
    reduce_scalar,
    register_sync_type,
    sync_object,
)

from .adaptive import (
    AdaptiveClippedGradAux,
    AdaptiveClipState,
    _adaptive_clip_norm_update,
    _sample_noisy_clipping_rate,
)
from .clipped_fun import ClippedFunAux
from .clipped_grad import ClippedGradAux
from .types import FixedClipState

__all__ = [
    "sync_clip_state",
    "sync_adaptive_clip_state",
    "sync_clipped_fun_aux",
    "sync_clipped_grad_aux",
    "sync_adaptive_clipped_grad_aux",
    "sync_aux",
]


def sync_clip_state(state: FixedClipState) -> FixedClipState:
    """Validate fixed clipping state is identical across ranks.

    Fixed clipping state must be identical across ranks. This function checks
    equality (within floating tolerance) and raises on mismatch.
    """
    if not is_distributed():
        return state
    if not isinstance(state, FixedClipState):
        raise TypeError(f"Expected FixedClipState, got {type(state)}")

    return sync_object(state, field_ops={"l2_norm_bound": "assert_equal"})


def sync_adaptive_clip_state(state: AdaptiveClipState) -> AdaptiveClipState:
    """Recompute adaptive clipping state from globally aggregated local counts.

    This function treats local adaptive updates as provisional and recomputes
    the effective global update from summed clipped counts.  It also sums
    ``batch_size`` across ranks so that the accounting layer receives the
    true global batch size.
    """
    if not is_distributed():
        return state

    # Use sync_object to sum the local counts and batch_size across ranks
    synced = sync_object(
        state,
        field_ops={
            "num_clipped": "sum",
            "total": "sum",
            "batch_size": "sum",
        },
    )

    # Recompute global clipping rate and clip_norm from aggregated counts
    global_rate = synced.num_clipped / max(1.0, synced.total)

    step_for_noise = max(0, synced.step - 1)
    noisy_global_rate = _sample_noisy_clipping_rate(
        global_rate,
        key=synced.key,
        step=step_for_noise,
        quantile_noise_multiplier=synced.quantile_noise_multiplier,
    )

    new_clip_norm = _adaptive_clip_norm_update(
        base_clip_norm=synced.base_clip_norm,
        noisy_clipping_rate=noisy_global_rate,
        target_quantile=synced.target_quantile,
        learning_rate=synced.learning_rate,
        clip_norm_min=synced.clip_norm_min,
        clip_norm_max=synced.clip_norm_max,
    )

    return replace(
        synced,
        clip_norm=float(new_clip_norm),
        clipping_rate=float(global_rate),
    )


def _split_aux_fields(aux: object) -> tuple[dict[str, object], dict[str, object]]:
    """Split NamedTuple fields into tensor-like and scalar/None groups."""
    tensor_fields: dict[str, object] = {}
    scalar_fields: dict[str, object] = {}

    for field_name in aux._fields:
        value = getattr(aux, field_name)
        if value is None or isinstance(value, (int, float)):
            scalar_fields[field_name] = value
        else:
            tensor_fields[field_name] = value

    return tensor_fields, scalar_fields


def sync_clipped_fun_aux(aux: ClippedFunAux) -> ClippedFunAux:
    """Synchronize ``ClippedFunAux`` by gathering per-example tensor fields."""
    if not is_distributed():
        return aux

    tensor_fields, scalar_fields = _split_aux_fields(aux)
    gathered = gather_pytree(tensor_fields) if tensor_fields else {}
    return ClippedFunAux(**{**gathered, **scalar_fields})


def sync_clipped_grad_aux(aux: ClippedGradAux) -> ClippedGradAux:
    """Synchronize ``ClippedGradAux`` and validate shared clipping norm."""
    if not is_distributed():
        return aux

    synced_fun_aux = sync_clipped_fun_aux(
        ClippedFunAux(
            loss_values=aux.loss_values,
            grad_norms=aux.grad_norms,
            clipped_grad_norms=aux.clipped_grad_norms,
            loss_aux=aux.loss_aux,
        )
    )

    assert_scalar_equal(aux.clipping_norm, name="clipping_norm")
    return ClippedGradAux(
        loss_values=synced_fun_aux.loss_values,
        grad_norms=synced_fun_aux.grad_norms,
        clipped_grad_norms=synced_fun_aux.clipped_grad_norms,
        loss_aux=synced_fun_aux.loss_aux,
        clipping_norm=aux.clipping_norm,
    )


def sync_adaptive_clipped_grad_aux(
    aux: AdaptiveClippedGradAux,
) -> AdaptiveClippedGradAux:
    """Synchronize ``AdaptiveClippedGradAux`` including global clipping rate."""
    if not is_distributed():
        return aux

    synced_fun_aux = sync_clipped_fun_aux(
        ClippedFunAux(
            loss_values=aux.loss_values,
            grad_norms=aux.grad_norms,
            clipped_grad_norms=aux.clipped_grad_norms,
            loss_aux=aux.loss_aux,
        )
    )

    clipping_rate = aux.clipping_rate
    if clipping_rate is None:
        global_clipping_rate = None
    else:
        local_n = 0.0
        if isinstance(aux.grad_norms, torch.Tensor):
            local_n = float(aux.grad_norms.numel())

        local_rate = float(clipping_rate)
        if local_n > 0:
            global_weighted_sum = reduce_scalar(local_rate * local_n, op="sum")
            global_total = reduce_scalar(local_n, op="sum")
            global_clipping_rate = global_weighted_sum / max(1.0, global_total)
        else:
            global_clipping_rate = reduce_scalar(local_rate, op="mean")

    return AdaptiveClippedGradAux(
        loss_values=synced_fun_aux.loss_values,
        grad_norms=synced_fun_aux.grad_norms,
        clipped_grad_norms=synced_fun_aux.clipped_grad_norms,
        loss_aux=synced_fun_aux.loss_aux,
        clipping_rate=global_clipping_rate,
    )


def sync_aux(
    aux: ClippedFunAux | ClippedGradAux | AdaptiveClippedGradAux,
) -> ClippedFunAux | ClippedGradAux | AdaptiveClippedGradAux:
    """Synchronize clipping auxiliary outputs across distributed ranks."""
    if isinstance(aux, AdaptiveClippedGradAux):
        return sync_adaptive_clipped_grad_aux(aux)
    if isinstance(aux, ClippedGradAux):
        return sync_clipped_grad_aux(aux)
    if isinstance(aux, ClippedFunAux):
        return sync_clipped_fun_aux(aux)
    raise TypeError(f"Unsupported aux type for sync_aux: {type(aux)}")


# Register all clipping types with the sync dispatcher
register_sync_type(FixedClipState, sync_clip_state)
register_sync_type(AdaptiveClipState, sync_adaptive_clip_state)
register_sync_type(ClippedFunAux, sync_clipped_fun_aux)
register_sync_type(ClippedGradAux, sync_clipped_grad_aux)
register_sync_type(AdaptiveClippedGradAux, sync_adaptive_clipped_grad_aux)
