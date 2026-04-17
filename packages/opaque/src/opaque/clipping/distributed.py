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
from opaque.random import fold_in, generator_from_key
from opaque.utils.per_group import PerGroup

from .adaptive import (
    AdaptiveClippedGradAux,
    AdaptiveClipState,
    _adaptive_clipping_norm_update,
    _sample_noisy_clipping_rate,
)
from .auto import AutoClippedGradAux, AutoClipState
from .auto_data import DataDependentAutoClipState
from .clipped_fun import ClippedFunAux
from .clipped_grad import ClippedGradAux
from .types import FixedClipState

__all__ = [
    "sync_clip_state",
    "sync_adaptive_clip_state",
    "sync_auto_clip_state",
    "sync_clipped_fun_aux",
    "sync_clipped_grad_aux",
    "sync_adaptive_clipped_grad_aux",
    "sync_auto_clipped_grad_aux",
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
    ``next_clipping_norm``.

    When ``_num_clipped`` is a dict (per-group adaptive clipping), each
    group's count is summed independently and the per-group thresholds
    are recomputed.
    """
    if not is_distributed():
        return state

    is_per_group = isinstance(state._num_clipped, dict)

    if is_per_group:
        # Per-group path: sync batch_size and each group's count separately.
        global_batch_size = reduce_scalar(float(state._batch_size), op="sum")

        global_num_clipped: dict[str, float] = {}
        for gname, local_count in state._num_clipped.items():
            global_num_clipped[gname] = reduce_scalar(local_count, op="sum")

        # Also assert normalize_by is equal across ranks
        reduce_scalar(float(state.normalize_by), op="mean")  # validation

        if global_batch_size == 0:
            return replace(state, _batch_size=global_batch_size)

        # Recompute per-group thresholds from globally aggregated counts
        current_pg = state.clipping_norm
        step_for_noise = max(0, state.step - 1)
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
            next_clipping_norm=new_clipping_norm,
            _num_clipped=global_num_clipped,
            _batch_size=global_batch_size,
        )

    # --- Scalar path (original) ---
    synced = sync_object(
        state,
        field_ops={
            "_num_clipped": "sum",
            "_batch_size": "sum",
            "normalize_by": "assert_equal",
        },
    )

    # No data on any rank — preserve clipping_norm, skip adaptation entirely
    if synced._batch_size == 0:
        return synced

    # Recompute global clipping rate from aggregated counts.
    global_rate = synced._num_clipped / max(1.0, synced._batch_size)

    step_for_noise = max(0, synced.step - 1)
    noisy_global_rate = _sample_noisy_clipping_rate(
        global_rate,
        key=synced._rng_key,
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
    scalar_fields["clipping_rate"] = _sync_clipping_rate(aux.clipping_rate, aux.norms)
    scalar_fields["batch_size"] = _sync_batch_size(aux.batch_size)

    return type(aux)(**{**gathered, **scalar_fields})


def _sync_clipping_rate(
    clipping_rate: float | None,
    norms: torch.Tensor | None,
) -> float | None:
    """Compute global clipping rate as weighted average across ranks.

    Each rank computes ``rate = num_clipped / batch_size``.  The weighted
    average ``sum(rate_i * n_i) / sum(n_i)`` yields the exact global rate.
    """
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


def sync_data_dependent_auto_clip_state(
    state: DataDependentAutoClipState,
) -> DataDependentAutoClipState:
    """Validate data-dependent AUTO-S state across ranks.

    Asserts ``clipping_norm`` (the safety clip) matches. The
    ``last_threshold`` may differ across ranks (each rank has a
    different local batch).
    """
    if not is_distributed():
        return state
    if not isinstance(state, DataDependentAutoClipState):
        raise TypeError(
            f"Expected DataDependentAutoClipState, got {type(state)}"
        )
    return sync_object(state, field_ops={"clipping_norm": "assert_equal"})


def sync_auto_clip_state(state: AutoClipState) -> AutoClipState:
    """Validate AUTO-S clipping state is identical across ranks.

    AUTO-S state is fixed (no adaptive counts), so sync just validates
    equality — same as ``sync_clip_state`` for ``FixedClipState``.
    """
    if not is_distributed():
        return state
    if not isinstance(state, AutoClipState):
        raise TypeError(f"Expected AutoClipState, got {type(state)}")
    return sync_object(state, field_ops={"clipping_norm": "assert_equal"})


def sync_auto_clipped_grad_aux(
    aux: AutoClippedGradAux,
) -> AutoClippedGradAux:
    """Synchronize ``AutoClippedGradAux`` across distributed ranks."""
    if not is_distributed():
        return aux
    return sync_clipped_grad_aux(aux)


def sync_aux(
    aux: ClippedFunAux | ClippedGradAux | AdaptiveClippedGradAux | AutoClippedGradAux,
) -> ClippedFunAux | ClippedGradAux | AdaptiveClippedGradAux | AutoClippedGradAux:
    """Synchronize clipping auxiliary outputs across distributed ranks."""
    if isinstance(aux, AutoClippedGradAux):
        return sync_auto_clipped_grad_aux(aux)
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
register_sync_type(AutoClipState, sync_auto_clip_state)
register_sync_type(DataDependentAutoClipState, sync_data_dependent_auto_clip_state)
register_sync_type(ClippedFunAux, sync_clipped_fun_aux)
register_sync_type(ClippedGradAux, sync_clipped_grad_aux)
register_sync_type(AdaptiveClippedGradAux, sync_adaptive_clipped_grad_aux)
register_sync_type(AutoClippedGradAux, sync_auto_clipped_grad_aux)
