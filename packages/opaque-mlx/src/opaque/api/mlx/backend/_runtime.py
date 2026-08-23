"""MLX implementations of optional runtime primitives."""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np

import mlx.core as mx
from opaque.api.engine import runtime
from opaque.api.mlx import distributed as mlx_distributed
from opaque.api.mlx.backend._core import _MLX

_MAX_OBJECT_BYTES = 64 * 1024 * 1024


def _group() -> Any | None:
    return mlx_distributed._registered_group()


def _world_size(group: Any | None = None) -> int:
    group = _group() if group is None else group
    return 1 if group is None else mlx_distributed._group_size(group)


def _validate_reduce_op(op: runtime.ReduceOp) -> runtime.ReduceOp:
    try:
        return runtime.ReduceOp(op)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid reduction operation: {op!r}. Must be one of: "
            f"{[item.value for item in runtime.ReduceOp.__members__.values()]}"
        ) from error


def _all_gather_object(value: Any, group: Any) -> list[Any]:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > _MAX_OBJECT_BYTES:
        raise ValueError(
            f"Distributed object payload exceeds {_MAX_OBJECT_BYTES} byte limit."
        )

    lengths = mx.distributed.all_gather(
        mx.array([len(payload)], dtype=mx.int32), group=group
    )
    mx.eval(lengths)
    lengths_array = np.asarray(lengths)
    expected_world_size = _world_size(group)
    if lengths_array.ndim != 1 or lengths_array.size != expected_world_size:
        raise RuntimeError("MLX object collective returned invalid length metadata.")
    payload_lengths = [int(length) for length in lengths_array]
    if any(length < 0 or length > _MAX_OBJECT_BYTES for length in payload_lengths):
        raise ValueError("MLX object collective received an invalid payload length.")

    padded_length = max(payload_lengths, default=0)
    padded_payload = np.zeros((padded_length,), dtype=np.uint8)
    padded_payload[: len(payload)] = np.frombuffer(payload, dtype=np.uint8)
    gathered = mx.distributed.all_gather(mx.array(padded_payload), group=group)
    mx.eval(gathered)
    gathered_array = np.asarray(gathered)
    if (
        gathered_array.ndim != 1
        or gathered_array.size != expected_world_size * padded_length
    ):
        raise RuntimeError("MLX object collective returned an invalid payload shape.")

    result = []
    for index, length in enumerate(payload_lengths):
        start = index * padded_length
        try:
            result.append(
                pickle.loads(gathered_array[start : start + length].tobytes())
            )
        except (pickle.UnpicklingError, EOFError, ValueError, TypeError) as error:
            raise ValueError(
                "MLX object collective received an invalid payload."
            ) from error
    return result


@_MLX.implements(runtime.distributed_initialized)
def distributed_is_initialized() -> bool:
    return _group() is not None


@_MLX.implements(runtime.distributed_rank)
def distributed_rank() -> int:
    group = _group()
    return 0 if group is None else mlx_distributed._group_rank(group)


@_MLX.implements(runtime.distributed_world_size)
def distributed_world_size() -> int:
    return _world_size()


@_MLX.implements(runtime.distributed_all_reduce)
def distributed_all_reduce(
    value: object, op: runtime.ReduceOp = runtime.ReduceOp.SUM
) -> object:
    operation = _validate_reduce_op(op)
    if isinstance(value, bool) or not isinstance(value, (mx.array, float, int)):
        raise TypeError(
            "MLX distributed reductions require an array, float, or int; "
            f"got {type(value).__name__}."
        )

    group = _group()
    if group is None:
        return mx.array(value) if isinstance(value, mx.array) else value
    if not isinstance(value, mx.array):
        values = _all_gather_object(value, group)
        if operation == runtime.ReduceOp.SUM:
            return sum(values)
        if operation == runtime.ReduceOp.MEAN:
            return sum(values) / len(values)
        if operation == runtime.ReduceOp.MIN:
            return min(values)
        if operation == runtime.ReduceOp.MAX:
            return max(values)
        return _product(values)

    if operation == runtime.ReduceOp.SUM:
        return mx.distributed.all_sum(value, group=group)
    if operation == runtime.ReduceOp.MEAN:
        return mx.distributed.all_sum(value, group=group) / _world_size(group)
    if operation == runtime.ReduceOp.MIN:
        return mx.distributed.all_min(value, group=group)
    if operation == runtime.ReduceOp.MAX:
        return mx.distributed.all_max(value, group=group)

    expanded = mx.expand_dims(value, axis=0)
    return mx.prod(mx.distributed.all_gather(expanded, group=group), axis=0)


def _product(values: list[float | int]) -> float | int:
    product: float | int = 1
    for value in values:
        product *= value
    return product


@_MLX.implements(runtime.distributed_all_gather_object)
def distributed_all_gather_object(value: Any) -> list[Any]:
    group = _group()
    return [value] if group is None else _all_gather_object(value, group)


@_MLX.implements(runtime.distributed_all_gather)
def distributed_all_gather(value: mx.array, *, axis: int = 0) -> mx.array:
    if not isinstance(value, mx.array):
        raise TypeError(
            f"MLX distributed array gathering requires an MLX array, got {type(value).__name__}."
        )
    group = _group()
    if group is None:
        return mx.array(value)

    if value.ndim == 0:
        if axis not in (0, -1):
            raise IndexError("axis is out of bounds for a scalar array")
        return mx.distributed.all_gather(mx.reshape(value, (1,)), group=group)
    if not -value.ndim <= axis < value.ndim:
        raise IndexError(
            f"axis {axis} is out of bounds for an array with {value.ndim} dimensions"
        )
    axis %= value.ndim
    metadata = _all_gather_object(
        (str(value.dtype), value.ndim, tuple(value.shape)), group
    )
    dtypes = [item[0] for item in metadata]
    ranks = [item[1] for item in metadata]
    shapes = [item[2] for item in metadata]
    if any(dtype != dtypes[0] for dtype in dtypes[1:]):
        raise TypeError("Distributed array gathering requires matching dtypes.")
    if any(rank != ranks[0] for rank in ranks[1:]):
        raise ValueError("Distributed array gathering requires matching array ranks.")
    reference_shape = shapes[0]
    if any(
        size != reference_shape[dimension]
        for shape in shapes[1:]
        for dimension, size in enumerate(shape)
        if dimension != axis
    ):
        raise ValueError(
            "Distributed array gathering requires matching non-concatenated dimensions."
        )

    moved = mx.moveaxis(value, axis, 0)
    return mx.moveaxis(mx.distributed.all_gather(moved, group=group), 0, axis)


@_MLX.implements(runtime.distributed_barrier)
def distributed_barrier(name: str | None = None) -> None:
    del name
    group = _group()
    if group is not None:
        marker = mx.distributed.all_sum(mx.zeros((1,), dtype=mx.int32), group=group)
        mx.eval(marker)


@_MLX.implements(runtime.synchronize)
def synchronize(device: Any = None) -> None:
    mx.synchronize(device)


@_MLX.implements(runtime.memory_stats)
def memory_stats(device: Any = None) -> runtime.MemoryStats:
    del device
    return runtime.MemoryStats(
        active_bytes=int(mx.get_active_memory()),
        cached_bytes=None,
        peak_active_bytes=int(mx.get_peak_memory()),
        capacity_bytes=None,
    )


@_MLX.implements(runtime.clear_memory_cache)
def clear_memory_cache(device: Any = None) -> None:
    del device
    mx.clear_cache()


@_MLX.implements(runtime.peak_memory_trackable)
def peak_memory_trackable(device: Any = None) -> bool:
    del device
    return hasattr(mx, "reset_peak_memory")


__all__: list[str] = []
