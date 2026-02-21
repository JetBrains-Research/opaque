"""Distributed synchronization helpers for clipping components.

This module contains clipping-specific distributed logic. Core collectives and
generic utilities remain in ``opaque.distributed``.
"""

from __future__ import annotations

from dataclasses import replace

from opaque.distributed import (
    assert_scalar_equal,
    gather_pytree,
    is_distributed,
    reduce_scalar,
)

from .adaptive import (
    AdaptiveClippedGradAux,
    AdaptiveClipState,
    _adaptive_clip_norm_update,
    _sample_noisy_clipping_rate,
)
from .types import FixedClipState

__all__ = [
    "sync_clip_state",
    "sync_adaptive_clip_state",
    "sync_adaptive_clipped_grad_aux",
]


def sync_clip_state(state: FixedClipState) -> FixedClipState:
    """Synchronize fixed clipping state and fail on mismatches.

    Fixed clipping state must be identical across ranks. This function checks
    equality (within floating tolerance) rather than averaging values.
    """
    if not is_distributed():
        return state
    if not isinstance(state, FixedClipState):
        raise TypeError(f"Expected FixedClipState, got {type(state)}")

    assert_scalar_equal(state.l2_norm_bound, name="FixedClipState.l2_norm_bound")
    return state


def sync_adaptive_clip_state(state: AdaptiveClipState) -> AdaptiveClipState:
    """Recompute adaptive clipping state from globally aggregated local counts.

    This function treats local adaptive updates as provisional and recomputes
    the effective global update from summed clipped counts.  It also sums
    ``batch_size`` across ranks so that the accounting layer receives the
    true global batch size.
    """
    if not is_distributed():
        return state

    global_num_clipped = reduce_scalar(float(state.num_clipped), op="sum")
    global_total = reduce_scalar(float(state.total), op="sum")
    global_batch_size = int(reduce_scalar(float(state.batch_size), op="sum"))
    global_rate = global_num_clipped / max(1.0, global_total)

    step_for_noise = max(0, state.step - 1)
    noisy_global_rate = _sample_noisy_clipping_rate(
        global_rate,
        key=state.key,
        step=step_for_noise,
        quantile_noise_multiplier=state.quantile_noise_multiplier,
    )

    new_clip_norm = _adaptive_clip_norm_update(
        base_clip_norm=state.base_clip_norm,
        noisy_clipping_rate=noisy_global_rate,
        target_quantile=state.target_quantile,
        learning_rate=state.learning_rate,
        clip_norm_min=state.clip_norm_min,
        clip_norm_max=state.clip_norm_max,
    )

    return replace(
        state,
        clip_norm=float(new_clip_norm),
        clipping_rate=float(global_rate),
        batch_size=global_batch_size,
    )


def sync_adaptive_clipped_grad_aux(
    aux: AdaptiveClippedGradAux,
) -> AdaptiveClippedGradAux:
    """Gather adaptive clipping auxiliary outputs across ranks."""
    if not is_distributed():
        return aux

    gathered = gather_pytree(
        {
            "loss_values": aux.loss_values,
            "grad_norms": aux.grad_norms,
            "clipped_grad_norms": aux.clipped_grad_norms,
            "loss_aux": aux.loss_aux,
        }
    )
    return AdaptiveClippedGradAux(
        loss_values=gathered.get("loss_values"),
        grad_norms=gathered.get("grad_norms"),
        clipped_grad_norms=gathered.get("clipped_grad_norms"),
        loss_aux=gathered.get("loss_aux"),
        clipping_rate=aux.clipping_rate,
    )
