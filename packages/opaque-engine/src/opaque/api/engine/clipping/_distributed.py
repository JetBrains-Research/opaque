"""Distributed synchronization helpers for core clipping components.

Implements all-reduce/all-gather patterns for the algorithm-agnostic
fixed clipping state and clipping auxiliary outputs. DP-SGD-specific
adaptive clipping sync lives in :mod:`opaque.dpsgd.clipping.distributed`
and self-registers via :func:`opaque.distributed.register_sync_type`.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch
import torch.distributed as dist

from opaque.api.engine.distributed import is_distributed
from opaque.api.engine.distributed._state import (
    get_world_size,
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


def _cpu_payload(value: Any) -> Any:
    """Detach tensors to CPU for ``all_gather_object`` pickling."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_payload(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_payload(v) for v in value)
    return value


def _infer_device(value: Any) -> torch.device:
    if isinstance(value, torch.Tensor):
        return value.device
    if isinstance(value, dict):
        for child in value.values():
            if isinstance(child, torch.Tensor):
                return child.device
    if isinstance(value, (list, tuple)):
        for child in value:
            if isinstance(child, torch.Tensor):
                return child.device
    return torch.device("cpu")


def _merge_gathered_values(values: list[Any], device: torch.device) -> Any:
    """Concatenate per-rank aux payloads; ``None`` ranks contribute nothing."""
    present = [v for v in values if v is not None]
    if not present:
        return None

    sample = present[0]
    if isinstance(sample, torch.Tensor):
        parts = [v.to(device) for v in values if isinstance(v, torch.Tensor)]
        return torch.cat(parts, dim=0) if parts else None

    if isinstance(sample, dict):
        keys: list[Any] = []
        seen: set[Any] = set()
        for payload in present:
            for key in payload:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        merged: dict[Any, Any] = {}
        for key in keys:
            per_rank = [
                None if payload is None else payload.get(key) for payload in values
            ]
            merged_value = _merge_gathered_values(per_rank, device)
            if merged_value is not None:
                merged[key] = merged_value
        return merged or None

    if isinstance(sample, (list, tuple)):
        length = len(sample)
        if any(len(v) != length for v in present):
            raise TypeError(
                "Distributed aux gather requires matching sequence lengths "
                f"across ranks; got {[len(v) for v in present]}"
            )
        merged_seq = [
            _merge_gathered_values([v[i] for v in present], device)
            for i in range(length)
        ]
        return type(sample)(merged_seq)

    raise TypeError(
        "Distributed aux gathering supports tensor / dict-of-tensor / None "
        f"leaves only; got {type(sample)}"
    )


def _gather_aux_fields(tensor_fields: dict[str, object]) -> dict[str, object]:
    """Gather schema tensor fields with one collective per field.

    Field-level ``all_gather_object`` (instead of ``tree_map`` over leaves)
    keeps the collective count identical when one rank has ``group_norms=None``
    (empty batch) and another has a per-group dict — a structure mismatch that
    ``gather_pytree`` cannot reconcile.
    """
    gathered: dict[str, object] = {}
    # Sorted keys → identical collective order on every rank.
    for name in sorted(tensor_fields):
        local = tensor_fields[name]
        payloads = [None] * get_world_size()
        dist.all_gather_object(payloads, _cpu_payload(local))
        gathered[name] = _merge_gathered_values(payloads, _infer_device(local))
    return gathered


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
    gathered = _gather_aux_fields(tensor_fields)

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
    gathered = _gather_aux_fields(tensor_fields)

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
