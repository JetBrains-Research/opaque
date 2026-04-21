"""Distributed synchronization helpers for core clipping components.

Implements all-reduce/all-gather patterns for the algorithm-agnostic
fixed clipping state and clipping auxiliary outputs. DP-SGD-specific
adaptive clipping sync lives in :mod:`opaque.dpsgd.clipping.distributed`
and self-registers via :func:`opaque.core.distributed.register_sync_type`.
"""

from __future__ import annotations

import dataclasses

import torch

from opaque.core.distributed import (
    gather_pytree,
    is_distributed,
    reduce_scalar,
    register_sync_type,
    sync_object,
)

from .clipped_fun import ClippedFunAux
from .clipped_grad import ClippedGradAux
from .types import FixedClipState

__all__ = [
    "sync_clip_state",
    "sync_clipped_fun_aux",
    "sync_clipped_grad_aux",
    "sync_aux",
]


def sync_clip_state(state: FixedClipState) -> FixedClipState:
    """Validate fixed clipping state is identical across ranks."""
    if not is_distributed():
        return state
    if not isinstance(state, FixedClipState):
        raise TypeError(f"Expected FixedClipState, got {type(state)}")

    return sync_object(state, field_ops={"clipping_norm": "assert_equal"})


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


def sync_clipped_fun_aux(aux: ClippedFunAux) -> ClippedFunAux:
    """Synchronize ``ClippedFunAux`` by gathering per-example tensor fields."""
    if not is_distributed():
        return aux

    tensor_fields, scalar_fields = _split_aux_fields(aux)
    gathered = gather_pytree(tensor_fields) if tensor_fields else {}

    scalar_fields["clipping_rate"] = _sync_clipping_rate(aux.clipping_rate, aux.norms)
    scalar_fields["batch_size"] = _sync_batch_size(aux.batch_size)

    return type(aux)(**{**gathered, **scalar_fields})


def sync_clipped_grad_aux(aux: ClippedGradAux) -> ClippedGradAux:
    """Synchronize ``ClippedGradAux`` across distributed ranks.

    Handles any subclass of :class:`ClippedGradAux` (e.g. the DP-SGD
    ``AdaptiveClippedGradAux``) — the constructor of ``type(aux)`` is
    used to rebuild the synced instance.
    """
    if not is_distributed():
        return aux

    tensor_fields, scalar_fields = _split_aux_fields(aux)
    gathered = gather_pytree(tensor_fields) if tensor_fields else {}

    scalar_fields["clipping_rate"] = _sync_clipping_rate(
        aux.clipping_rate, aux.grad_norms
    )
    scalar_fields["batch_size"] = _sync_batch_size(aux.batch_size)

    return type(aux)(**{**gathered, **scalar_fields})


def sync_aux(
    aux: ClippedFunAux | ClippedGradAux,
) -> ClippedFunAux | ClippedGradAux:
    """Synchronize clipping auxiliary outputs across distributed ranks."""
    if isinstance(aux, ClippedGradAux):
        return sync_clipped_grad_aux(aux)
    if isinstance(aux, ClippedFunAux):
        return sync_clipped_fun_aux(aux)
    raise TypeError(f"Unsupported aux type for sync_aux: {type(aux)}")


register_sync_type(FixedClipState, sync_clip_state)
register_sync_type(ClippedFunAux, sync_clipped_fun_aux)
register_sync_type(ClippedGradAux, sync_clipped_grad_aux)
