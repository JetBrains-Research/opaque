"""Distributed synchronization helpers for core clipping components.

Implements all-reduce/all-gather patterns for the algorithm-agnostic
fixed clipping state and clipping auxiliary outputs. DP-SGD-specific
adaptive clipping sync lives in :mod:`opaque.dpsgd.clipping.distributed`
and self-registers via :func:`opaque.distributed.register_sync_type`.
"""

from __future__ import annotations

import dataclasses

import torch

from opaque.api.engine.distributed import is_distributed
from opaque.api.engine.distributed._state import (
    gather_pytree,
    reduce_scalar,
    register_sync_type,
)

from ._clipped_fun import ClippedFunAux, FixedClipState
from ._clipped_grad import ClippedGradAux

__all__ = [
    "sync_aux",
    "sync_clip_state",
    "sync_clipped_fun_aux",
    "sync_clipped_grad_aux",
]


def sync_clip_state(state: FixedClipState) -> FixedClipState:
    """Synchronize fixed clipping marker state."""
    if not is_distributed():
        return state
    if not isinstance(state, FixedClipState):
        raise TypeError(f"Expected FixedClipState, got {type(state)}")

    return state


# Scalar aux fields reduced with a fixed all-reduce sequence. Everything else
# is treated as gather-able (tensors / nested tensor pytrees / None placeholders)
# so ranks with empty Poisson batches still walk the same collective schedule.
_SCALAR_AUX_FIELDS = frozenset({"clipping_rate", "batch_size"})


def _split_aux_fields(aux) -> tuple[dict[str, object], dict[str, object]]:
    """Split dataclass fields by schema, not by runtime value.

    Classifying by ``isinstance`` / ``is None`` made empty-batch ranks drop
    tensor fields from the gather map while non-empty ranks kept them, so the
    two sides issued different collective sequences and permanently desynced
    the process group.
    """
    tensor_fields: dict[str, object] = {}
    scalar_fields: dict[str, object] = {}

    for f in dataclasses.fields(aux):
        value = getattr(aux, f.name)
        if f.name in _SCALAR_AUX_FIELDS:
            scalar_fields[f.name] = value
        else:
            tensor_fields[f.name] = value

    return tensor_fields, scalar_fields


def _sync_clipping_rate(
    clipping_rate: float | None,
    norms: torch.Tensor | None,
) -> float | None:
    """Compute global clipping rate as a size-weighted average across ranks.

    Always issues the same two ``reduce_scalar`` collectives. An empty local
    batch contributes weight 0; branching on ``local_n > 0`` previously made
    empty ranks issue one all-reduce while non-empty ranks issued two, which
    desynchronized the process group after a single empty Poisson draw.
    """
    if clipping_rate is None:
        return None

    local_n = float(norms.numel()) if isinstance(norms, torch.Tensor) else 0.0
    local_rate = float(clipping_rate)
    global_weighted_sum = reduce_scalar(local_rate * local_n, op="sum")
    global_total = reduce_scalar(local_n, op="sum")
    if global_total <= 0.0:
        return 0.0
    return global_weighted_sum / global_total


def _sync_batch_size(batch_size: int) -> int:
    """Sum batch_size across ranks."""
    return int(reduce_scalar(float(batch_size), op="sum"))


def sync_clipped_fun_aux(aux: ClippedFunAux) -> ClippedFunAux:
    """Synchronize ``ClippedFunAux`` by gathering per-example tensor fields."""
    if not is_distributed():
        return aux

    tensor_fields, scalar_fields = _split_aux_fields(aux)
    # Always gather the schema tensor map (possibly all-None / empty) so every
    # rank issues the same all_gather_object sequence.
    gathered = gather_pytree(tensor_fields)

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
    gathered = gather_pytree(tensor_fields)

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
