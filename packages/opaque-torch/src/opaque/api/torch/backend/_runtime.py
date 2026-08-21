"""Torch implementations of optional engine runtime capabilities."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from opaque.api.engine import runtime
from opaque.api.engine.backend import KnownBackend
from opaque.api.engine.primitive import BackendProvider

_TORCH = BackendProvider(KnownBackend.TORCH)


@_TORCH.implements(runtime.distributed_initialized)
def distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


@_TORCH.implements(runtime.distributed_rank)
def distributed_rank() -> int:
    return dist.get_rank() if distributed_is_initialized() else 0


@_TORCH.implements(runtime.distributed_world_size)
def distributed_world_size() -> int:
    return dist.get_world_size() if distributed_is_initialized() else 1


@_TORCH.implements(runtime.distributed_all_reduce)
def distributed_all_reduce(
    value: object, op: runtime.ReduceOp = runtime.ReduceOp.SUM
) -> object:
    operations = {
        "sum": dist.ReduceOp.SUM,
        "mean": dist.ReduceOp.SUM,
        "max": dist.ReduceOp.MAX,
        "min": dist.ReduceOp.MIN,
        "product": dist.ReduceOp.PRODUCT,
    }
    try:
        reduce_op = operations[op]
    except KeyError as error:
        raise ValueError(
            f"Invalid reduction operation: {op}. Must be one of: {list(operations)}"
        ) from error
    if isinstance(value, bool) or not isinstance(value, (torch.Tensor, float, int)):
        raise TypeError(
            "Torch distributed reductions require a tensor, float, or int; "
            f"got {type(value).__name__}."
        )
    scalar = not isinstance(value, torch.Tensor)
    integer = isinstance(value, int)
    if scalar:
        device = None
        if distributed_is_initialized() and dist.get_backend() == "nccl":
            device = torch.device("cuda", torch.cuda.current_device())
        # A Python float *is* a float64, so reduce it as one.  Reducing at
        # ``torch.get_default_dtype()`` would let a process-global setting
        # decide a DP-relevant scalar's precision: under a bfloat16 default
        # ``3.14159265358979`` comes back as ``3.140625``, and even the
        # float32 default cannot hold a sum past 2**24 exactly.
        reduced = torch.tensor(
            value,
            dtype=torch.int64 if integer else torch.float64,
            device=device,
        )
    else:
        reduced = value.clone()
    if distributed_is_initialized():
        dist.all_reduce(reduced, op=reduce_op)
        if op == runtime.ReduceOp.MEAN and not (scalar and integer):
            reduced = reduced / distributed_world_size()
    if scalar:
        result = reduced.item()
        if integer and op == runtime.ReduceOp.MEAN:
            # Divide the exact int64 sum in Python (float64) — tensor true
            # division would round through float32 and lose exactness for
            # sums past 2**24.
            return result / distributed_world_size()
        return result
    return reduced


@_TORCH.implements(runtime.distributed_all_gather_object)
def distributed_all_gather_object(value: Any) -> list[Any]:
    if not distributed_is_initialized():
        return [value]
    gathered: list[Any] = [None] * distributed_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


@_TORCH.implements(runtime.distributed_barrier)
def distributed_barrier(name: str | None = None) -> None:
    del name
    if distributed_is_initialized():
        dist.barrier()


@_TORCH.implements(runtime.distributed_all_gather)
def distributed_all_gather(value: torch.Tensor, *, axis: int = 0) -> torch.Tensor:
    if not distributed_is_initialized():
        return value.clone()
    if value.dim() == 0:
        if axis not in (0, -1):
            raise IndexError("axis is out of bounds for a scalar array")
        value = value.unsqueeze(0)
        axis = 0
    if not -value.dim() <= axis < value.dim():
        raise IndexError(
            f"axis {axis} is out of bounds for an array with {value.dim()} dimensions"
        )
    axis %= value.dim()
    local_metadata = (value.dtype, value.dim(), tuple(value.shape))
    metadata: list[Any] = [None] * distributed_world_size()
    dist.all_gather_object(metadata, local_metadata)
    dtypes = [item[0] for item in metadata]
    ranks = [item[1] for item in metadata]
    shapes = [item[2] for item in metadata]
    if any(dtype != dtypes[0] for dtype in dtypes[1:]):
        raise TypeError("Distributed array gathering requires matching dtypes.")
    if any(rank != ranks[0] for rank in ranks[1:]):
        raise ValueError("Distributed array gathering requires matching tensor ranks.")
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
    rank_lengths = [shape[axis] for shape in shapes]
    max_length = max(rank_lengths)
    padded_shape = list(value.shape)
    padded_shape[axis] = max_length
    padded = value.new_zeros(padded_shape)
    local_slice = [slice(None)] * value.dim()
    local_slice[axis] = slice(0, value.shape[axis])
    padded[tuple(local_slice)] = value
    gathered = [torch.empty_like(padded) for _ in rank_lengths]
    dist.all_gather(gathered, padded.contiguous())
    return torch.cat(
        [
            tensor.narrow(axis, 0, length)
            for tensor, length in zip(gathered, rank_lengths, strict=True)
        ],
        dim=axis,
    )


@_TORCH.implements(runtime.memory_stats)
def profiling_memory_stats(device: Any = None) -> runtime.MemoryStats:
    device = _normalize_device(device)
    if device.type == "cuda":
        active_bytes = torch.cuda.memory_allocated(device)
        reserved_bytes = torch.cuda.memory_reserved(device)
        return runtime.MemoryStats(
            active_bytes=active_bytes,
            cached_bytes=max(reserved_bytes - active_bytes, 0),
            peak_active_bytes=torch.cuda.max_memory_allocated(device),
            capacity_bytes=torch.cuda.get_device_properties(device).total_memory,
        )
    if device.type == "mps":
        active_bytes = torch.mps.current_allocated_memory()
        driver_bytes = torch.mps.driver_allocated_memory()
        return runtime.MemoryStats(
            active_bytes=active_bytes,
            cached_bytes=max(driver_bytes - active_bytes, 0),
            peak_active_bytes=driver_bytes,
            capacity_bytes=torch.mps.recommended_max_memory(),
        )
    return runtime.MemoryStats()


@_TORCH.implements(runtime.reset_peak_memory)
def profiling_reset_peak_memory(device: Any = None) -> None:
    device = _normalize_device(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "mps":
        # No peak counter on MPS; releasing cached-but-unused blocks resets
        # the driver's reserved high-water mark. Heavier than CUDA's counter
        # reset — call between measured configs, not every step.
        torch.mps.empty_cache()
    # CPU: no-op — process RSS has no resettable high-water counter.


@_TORCH.implements(runtime.peak_memory_trackable)
def profiling_peak_memory_trackable(device: Any = None) -> bool:
    # Only CUDA exposes a cheap per-step peak counter; the MPS "reset" is
    # an allocator cache flush and CPU has no resettable counter at all.
    return _normalize_device(device).type == "cuda"


@_TORCH.implements(runtime.clear_memory_cache)
def profiling_empty_cache(device: Any = None) -> None:
    device = _normalize_device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


@_TORCH.implements(runtime.synchronize)
def profiling_synchronize(device: Any = None) -> None:
    device = _normalize_device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _normalize_device(device: Any = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@_TORCH.implements(runtime.trace_scope)
def profiling_trace_scope(label: str) -> Any:
    return torch.autograd.profiler.record_function(label)


__all__: list[str] = []
