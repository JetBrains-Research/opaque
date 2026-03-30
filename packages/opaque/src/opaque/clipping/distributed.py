"""Distributed synchronization helpers for clipping components.

This module contains clipping-specific distributed logic. Core collectives and
generic utilities remain in ``opaque.distributed``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace

import torch

from opaque.distributed import (
    gather_pytree,
    is_distributed,
    reduce_scalar,
    register_sync_type,
    sync_object,
)

from .adaptive import (
    AdaptiveClippedGradAux,
    AdaptiveClipState,
    _adaptive_clipping_norm_update,
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

    return sync_object(state, field_ops={"clipping_norm": "assert_equal"})


def sync_adaptive_clip_state(state: AdaptiveClipState) -> AdaptiveClipState:
    """Recompute adaptive clipping state from globally aggregated local counts.

    Sums ``_num_clipped`` and ``_batch_size`` across ranks, recomputes the
    global clipping rate, applies quantile noise, and updates
    ``next_clipping_norm``.  ``normalize_by`` (data-independent constant)
    is validated to be equal across ranks and used as the fraction
    denominator when > 1.
    """
    if not is_distributed():
        return state

    synced = sync_object(
        state,
        field_ops={
            "_num_clipped": "sum",
            "_batch_size": "sum",
            "normalize_by": "assert_equal",
        },
    )

    # Recompute global clipping rate from aggregated counts.
    denominator = (
        synced.normalize_by if synced.normalize_by > 1.0
        else max(1.0, synced._batch_size)
    )
    global_rate = synced._num_clipped / max(1.0, denominator)

    step_for_noise = max(0, synced.step - 1)
    noisy_global_rate = _sample_noisy_clipping_rate(
        global_rate,
        key=synced._key,
        step=step_for_noise,
        fraction_noise_std=synced._fraction_noise_std,
    )

    new_clipping_norm = _adaptive_clipping_norm_update(
        base_clipping_norm=synced.clipping_norm,
        noisy_clipping_rate=noisy_global_rate,
        target_quantile=synced._target_quantile,
        learning_rate=synced._learning_rate,
        clipping_norm_min=synced._clipping_norm_min,
        clipping_norm_max=synced._clipping_norm_max,
    )

    return replace(
        synced,
        next_clipping_norm=float(new_clipping_norm),
    )


def _split_aux_fields(aux) -> tuple[dict[str, object], dict[str, object]]:
    """Split dataclass fields into tensor-like and scalar/None groups."""
    tensor_fields: dict[str, object] = {}
    scalar_fields: dict[str, object] = {}

    for f in dataclasses.fields(aux):
        value = getattr(aux, f.name)
        if value is None or isinstance(value, (int, float)):
            scalar_fields[f.name] = value
        else:
            tensor_fields[f.name] = value

    return tensor_fields, scalar_fields


def sync_clipped_fun_aux(aux: ClippedFunAux) -> ClippedFunAux:
    """Synchronize ``ClippedFunAux`` by gathering per-example tensor fields."""
    if not is_distributed():
        return aux

    tensor_fields, scalar_fields = _split_aux_fields(aux)
    gathered = gather_pytree(tensor_fields) if tensor_fields else {}

    # Override scalar fields that need distributed sync
    scalar_fields["clipping_rate"] = _sync_clipping_rate(
        aux.clipping_rate, aux.norms
    )
    scalar_fields["batch_size"] = _sync_batch_size(aux.batch_size)

    return type(aux)(**{**gathered, **scalar_fields})


def _sync_clipping_rate(
    clipping_rate: float | None,
    norms: torch.Tensor | None,
) -> float | None:
    """Compute global clipping rate as weighted average across ranks."""
    if clipping_rate is None:
        return None

    local_n = 0.0
    if isinstance(norms, torch.Tensor):
        local_n = float(norms.numel())

    local_rate = float(clipping_rate)
    if local_n > 0:
        global_weighted_sum = reduce_scalar(local_rate * local_n, op="sum")
        global_total = reduce_scalar(local_n, op="sum")
        return global_weighted_sum / max(1.0, global_total)
    else:
        return reduce_scalar(local_rate, op="mean")


def _sync_batch_size(batch_size: int) -> int:
    """Sum batch_size across ranks."""
    return int(reduce_scalar(float(batch_size), op="sum"))


def sync_clipped_grad_aux(aux: ClippedGradAux) -> ClippedGradAux:
    """Synchronize ``ClippedGradAux`` across distributed ranks."""
    if not is_distributed():
        return aux

    tensor_fields, scalar_fields = _split_aux_fields(aux)
    gathered = gather_pytree(tensor_fields) if tensor_fields else {}

    scalar_fields["clipping_rate"] = _sync_clipping_rate(
        aux.clipping_rate, aux.grad_norms
    )
    scalar_fields["batch_size"] = _sync_batch_size(aux.batch_size)

    return type(aux)(**{**gathered, **scalar_fields})


def sync_adaptive_clipped_grad_aux(
    aux: AdaptiveClippedGradAux,
) -> AdaptiveClippedGradAux:
    """Synchronize ``AdaptiveClippedGradAux`` across distributed ranks."""
    if not is_distributed():
        return aux
    return sync_clipped_grad_aux(aux)


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
