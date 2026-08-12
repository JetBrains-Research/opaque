"""MLX implementations of optional engine runtime capabilities."""

from __future__ import annotations

import pickle
from functools import lru_cache
from typing import Any

import mlx.core as mx
from opaque.api.engine import runtime
from opaque.api.engine.backend import KnownBackend
from opaque.api.engine.primitive import BackendProvider

_MLX = BackendProvider(KnownBackend.MLX)


@_MLX.implements(runtime.synchronize)
def synchronize(device: object | None = None) -> None:
    mx.synchronize(device)


@_MLX.implements(runtime.memory_stats)
def memory_stats(device: object | None = None) -> runtime.MemoryStats:
    del device
    return runtime.MemoryStats(
        active_bytes=mx.get_active_memory(),
        cached_bytes=mx.get_cache_memory(),
        peak_active_bytes=mx.get_peak_memory(),
    )


@_MLX.implements(runtime.clear_memory_cache)
def clear_memory_cache(device: object | None = None) -> None:
    del device
    mx.clear_cache()


@_MLX.implements(runtime.reset_peak_memory)
def reset_peak_memory(device: object | None = None) -> None:
    del device
    mx.reset_peak_memory()


@lru_cache(maxsize=1)
def _global_group() -> Any:
    return mx.distributed.init(strict=False)


def _reduce_op(op: runtime.ReduceOp | str) -> runtime.ReduceOp:
    try:
        return runtime.ReduceOp(op)
    except ValueError as error:
        raise ValueError(
            f"Invalid reduction operation: {op}. Must be one of: "
            f"{[item.value for item in runtime.ReduceOp]}"
        ) from error


def _clone(value: Any) -> Any:
    return mx.add(value, mx.zeros_like(value))


@_MLX.implements(runtime.distributed_rank)
def distributed_rank() -> int:
    return _global_group().rank()


@_MLX.implements(runtime.distributed_world_size)
def distributed_world_size() -> int:
    return _global_group().size()


@_MLX.implements(runtime.distributed_all_reduce)
def distributed_all_reduce(
    value: object, op: runtime.ReduceOp = runtime.ReduceOp.SUM
) -> object:
    operation = _reduce_op(op)
    scalar = isinstance(value, (float, int)) and not isinstance(value, bool)
    if not scalar and not isinstance(value, mx.array):
        raise TypeError(
            "MLX distributed reductions require an array, float, or int; "
            f"got {type(value).__name__}."
        )
    local = (
        mx.array(value, dtype=mx.int64 if isinstance(value, int) else mx.float32)
        if scalar
        else value
    )
    group = _global_group()
    if group.size() == 1:
        reduced = _clone(local)
    elif operation == runtime.ReduceOp.PRODUCT:
        gathered = mx.distributed.all_gather(mx.expand_dims(local, axis=0), group=group)
        reduced = mx.prod(gathered.reshape((group.size(), *local.shape)), axis=0)
    else:
        collectives = {
            runtime.ReduceOp.SUM: mx.distributed.all_sum,
            runtime.ReduceOp.MEAN: mx.distributed.all_sum,
            runtime.ReduceOp.MIN: mx.distributed.all_min,
            runtime.ReduceOp.MAX: mx.distributed.all_max,
        }
        reduced = collectives[operation](local, group=group)
        if operation == runtime.ReduceOp.MEAN:
            reduced = reduced / group.size()
    if scalar:
        mx.eval(reduced)
        return reduced.item()
    return reduced


@_MLX.implements(runtime.distributed_all_gather_object)
def distributed_all_gather_object(value: Any) -> list[Any]:
    group = _global_group()
    world_size = group.size()
    if world_size == 1:
        return [value]

    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    lengths_array = mx.distributed.all_gather(
        mx.array([len(payload)], dtype=mx.uint32), group=group
    )
    mx.eval(lengths_array)
    lengths = [int(length) for length in lengths_array.tolist()]
    if len(lengths) != world_size:
        raise RuntimeError(
            "MLX object gathering returned an unexpected number of payload lengths."
        )
    width = max(lengths)
    padded = mx.array([*payload, *([0] * (width - len(payload)))], dtype=mx.uint8)
    gathered = mx.distributed.all_gather(padded, group=group).reshape(world_size, width)
    mx.eval(gathered)
    return [
        pickle.loads(bytes(gathered[rank, :length].tolist()))
        for rank, length in enumerate(lengths)
    ]


def _normalize_gather_axis(value: Any, axis: int) -> tuple[Any, int]:
    ndim = value.ndim
    if ndim == 0:
        if axis not in (0, -1):
            raise IndexError("axis is out of bounds for a scalar array")
        if distributed_world_size() == 1:
            return value, 0
        return mx.expand_dims(value, axis=0), 0
    if not -ndim <= axis < ndim:
        raise IndexError(
            f"axis {axis} is out of bounds for an array with {ndim} dimensions"
        )
    return value, axis % ndim


def _validate_gather_metadata(
    metadata: list[tuple[str, int, tuple[int, ...]]], axis: int
) -> list[int]:
    dtypes = [item[0] for item in metadata]
    ranks = [item[1] for item in metadata]
    shapes = [item[2] for item in metadata]
    if any(dtype != dtypes[0] for dtype in dtypes[1:]):
        raise TypeError("Distributed array gathering requires matching dtypes.")
    if any(rank != ranks[0] for rank in ranks[1:]):
        raise ValueError("Distributed array gathering requires matching array ranks.")
    reference = shapes[0]
    for shape in shapes[1:]:
        if any(
            size != reference[dimension]
            for dimension, size in enumerate(shape)
            if dimension != axis
        ):
            raise ValueError(
                "Distributed array gathering requires matching "
                "non-concatenated dimensions."
            )
    return [shape[axis] for shape in shapes]


@_MLX.implements(runtime.distributed_all_gather)
def distributed_all_gather(value: object, *, axis: int = 0) -> object:
    if not isinstance(value, mx.array):
        raise TypeError(
            "MLX distributed array gathering requires an MLX array; "
            f"got {type(value).__name__}."
        )
    original = value
    value, axis = _normalize_gather_axis(value, axis)
    group = _global_group()
    if group.size() == 1:
        return _clone(original)

    local_metadata = (str(value.dtype), value.ndim, tuple(value.shape))
    metadata = distributed_all_gather_object(local_metadata)
    rank_lengths = _validate_gather_metadata(metadata, axis)
    max_length = max(rank_lengths)
    if max_length == 0:
        return _clone(value)

    padding_shape = list(value.shape)
    padding_shape[axis] = max_length - value.shape[axis]
    padded = mx.concatenate(
        [value, mx.zeros(padding_shape, dtype=value.dtype)], axis=axis
    )
    moved = mx.moveaxis(padded, axis, 0)
    gathered = mx.distributed.all_gather(moved, group=group)
    trimmed = [
        gathered[rank * max_length : rank * max_length + length]
        for rank, length in enumerate(rank_lengths)
    ]
    return mx.moveaxis(mx.concatenate(trimmed, axis=0), 0, axis)


@_MLX.implements(runtime.distributed_barrier)
def distributed_barrier(name: str | None = None) -> None:
    del name
    group = _global_group()
    if group.size() > 1:
        sentinel = mx.distributed.all_sum(mx.array(0, dtype=mx.int32), group=group)
        mx.eval(sentinel)


__all__: list[str] = []
