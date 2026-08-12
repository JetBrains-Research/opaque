"""JAX implementations of optional engine runtime capabilities."""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
from opaque.api.engine import runtime
from opaque.api.engine.backend import KnownBackend
from opaque.api.engine.primitive import BackendProvider

_JAX = BackendProvider(KnownBackend.JAX)


def _available_stat(stats: dict[str, int], *keys: str) -> int | None:
    for key in keys:
        value = stats.get(key)
        if value is not None and value >= 0:
            return int(value)
    return None


@_JAX.implements(runtime.synchronize)
def synchronize(device: object | None = None) -> None:
    del device
    jax.effects_barrier()


@_JAX.implements(runtime.memory_stats)
def memory_stats(device: object | None = None) -> runtime.MemoryStats:
    selected_device = jax.devices()[0] if device is None else device
    native_stats = selected_device.memory_stats()
    if not native_stats:
        return runtime.MemoryStats()

    active_bytes = _available_stat(native_stats, "bytes_in_use")
    pool_bytes = _available_stat(native_stats, "pool_bytes", "bytes_reserved")
    cached_bytes = None
    if (
        active_bytes is not None
        and pool_bytes is not None
        and pool_bytes >= active_bytes
    ):
        cached_bytes = pool_bytes - active_bytes
    return runtime.MemoryStats(
        active_bytes=active_bytes,
        cached_bytes=cached_bytes,
        peak_active_bytes=_available_stat(native_stats, "peak_bytes_in_use"),
        capacity_bytes=_available_stat(
            native_stats, "bytes_limit", "bytes_reservable_limit"
        ),
    )


@_JAX.implements(runtime.trace_scope)
def trace_scope(label: str) -> Any:
    return jax.profiler.TraceAnnotation(label)


def _reduce_op(op: runtime.ReduceOp | str) -> runtime.ReduceOp:
    try:
        return runtime.ReduceOp(op)
    except ValueError as error:
        raise ValueError(
            f"Invalid reduction operation: {op}. Must be one of: "
            f"{[item.value for item in runtime.ReduceOp]}"
        ) from error


@_JAX.implements(runtime.distributed_rank)
def distributed_rank() -> int:
    return jax.process_index()


@_JAX.implements(runtime.distributed_world_size)
def distributed_world_size() -> int:
    return jax.process_count()


@_JAX.implements(runtime.distributed_all_reduce)
def distributed_all_reduce(
    value: object, op: runtime.ReduceOp = runtime.ReduceOp.SUM
) -> object:
    operation = _reduce_op(op)
    scalar = isinstance(value, (float, int)) and not isinstance(value, bool)
    if not scalar and not isinstance(value, jax.Array):
        raise TypeError(
            "JAX distributed reductions require an array, float, or int; "
            f"got {type(value).__name__}."
        )

    if scalar:
        local = np.asarray(value)
        if distributed_world_size() == 1:
            gathered = np.expand_dims(local, axis=0)
        else:
            gathered = np.asarray(multihost_utils.process_allgather(local, tiled=False))
        reducers = {
            runtime.ReduceOp.SUM: np.sum,
            runtime.ReduceOp.MEAN: np.mean,
            runtime.ReduceOp.MIN: np.min,
            runtime.ReduceOp.MAX: np.max,
            runtime.ReduceOp.PRODUCT: np.prod,
        }
        return reducers[operation](gathered, axis=0).item()

    if distributed_world_size() == 1:
        return jnp.array(value, copy=True)
    gathered = jnp.asarray(multihost_utils.process_allgather(value, tiled=False))
    reducers = {
        runtime.ReduceOp.SUM: jnp.sum,
        runtime.ReduceOp.MEAN: jnp.mean,
        runtime.ReduceOp.MIN: jnp.min,
        runtime.ReduceOp.MAX: jnp.max,
        runtime.ReduceOp.PRODUCT: jnp.prod,
    }
    return reducers[operation](gathered, axis=0)


@_JAX.implements(runtime.distributed_all_gather_object)
def distributed_all_gather_object(value: Any) -> list[Any]:
    world_size = distributed_world_size()
    if world_size == 1:
        return [value]

    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    lengths = np.asarray(
        multihost_utils.process_allgather(
            np.asarray([len(payload)], dtype=np.uint32), tiled=False
        )
    ).reshape(-1)
    if len(lengths) != world_size:
        raise RuntimeError(
            "JAX object gathering returned an unexpected number of payload lengths."
        )
    width = int(lengths.max())
    padded = np.zeros(width, dtype=np.uint8)
    padded[: len(payload)] = np.frombuffer(payload, dtype=np.uint8)
    gathered = np.asarray(
        multihost_utils.process_allgather(padded, tiled=False), dtype=np.uint8
    ).reshape(world_size, width)
    return [
        pickle.loads(gathered[rank, : int(length)].tobytes())
        for rank, length in enumerate(lengths)
    ]


def _normalize_gather_axis(value: jax.Array, axis: int) -> tuple[jax.Array, int]:
    ndim = value.ndim
    if ndim == 0:
        if axis not in (0, -1):
            raise IndexError("axis is out of bounds for a scalar array")
        if distributed_world_size() == 1:
            return value, 0
        return jnp.expand_dims(value, axis=0), 0
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


@_JAX.implements(runtime.distributed_all_gather)
def distributed_all_gather(value: object, *, axis: int = 0) -> object:
    if not isinstance(value, jax.Array):
        raise TypeError(
            "JAX distributed array gathering requires a JAX array; "
            f"got {type(value).__name__}."
        )
    original = value
    value, axis = _normalize_gather_axis(value, axis)
    if distributed_world_size() == 1:
        return jnp.array(original, copy=True)

    local_metadata = (str(value.dtype), value.ndim, tuple(value.shape))
    metadata = distributed_all_gather_object(local_metadata)
    rank_lengths = _validate_gather_metadata(metadata, axis)
    max_length = max(rank_lengths)
    if max_length == 0:
        return jnp.array(value, copy=True)

    pad_width = [(0, 0)] * value.ndim
    pad_width[axis] = (0, max_length - value.shape[axis])
    padded = jnp.pad(value, pad_width)
    gathered = jnp.asarray(multihost_utils.process_allgather(padded, tiled=False))
    trimmed = []
    for rank, length in enumerate(rank_lengths):
        slices = [slice(None)] * value.ndim
        slices[axis] = slice(0, length)
        trimmed.append(gathered[rank][tuple(slices)])
    return jnp.concatenate(trimmed, axis=axis)


@_JAX.implements(runtime.distributed_barrier)
def distributed_barrier(name: str | None = None) -> None:
    if distributed_world_size() > 1:
        multihost_utils.sync_global_devices(
            name or "opaque.runtime.distributed.barrier"
        )


__all__: list[str] = []
