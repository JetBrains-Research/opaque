"""Distributed synchronization helpers for clipping components.

This module contains clipping-specific distributed logic. Core collectives and
generic utilities remain in ``opaque.distributed``.
"""

from __future__ import annotations

from dataclasses import replace

from opaque.distributed import (
    gather_pytree,
    is_distributed,
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


def sync_aux(
    aux: ClippedFunAux | ClippedGradAux | AdaptiveClippedGradAux,
) -> ClippedFunAux | ClippedGradAux | AdaptiveClippedGradAux:
    """Gather per-example auxiliary outputs across ranks.

    Works with any of the clipping aux types (``ClippedFunAux``,
    ``ClippedGradAux``, ``AdaptiveClippedGradAux``).  Tensor fields are
    concatenated along the batch dimension; scalar and ``None`` fields are
    preserved as-is.

    Args:
        aux: Auxiliary NamedTuple from any clipping function.

    Returns:
        New aux of the same type with gathered tensor fields.
    """
    if not is_distributed():
        return aux

    # Separate tensor fields (to gather) from scalar/None fields (to keep)
    tensor_fields = {}
    scalar_fields = {}
    for field_name in aux._fields:
        value = getattr(aux, field_name)
        if value is None or isinstance(value, (int, float)):
            scalar_fields[field_name] = value
        else:
            tensor_fields[field_name] = value

    gathered = gather_pytree(tensor_fields) if tensor_fields else {}

    return type(aux)(**{**gathered, **scalar_fields})


# Register all clipping types with the sync dispatcher
register_sync_type(FixedClipState, sync_clip_state)
register_sync_type(AdaptiveClipState, sync_adaptive_clip_state)
register_sync_type(ClippedFunAux, sync_aux)
register_sync_type(ClippedGradAux, sync_aux)
register_sync_type(AdaptiveClippedGradAux, sync_aux)
