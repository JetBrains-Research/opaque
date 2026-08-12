"""Torch implementations of optional engine runtime capabilities."""

from __future__ import annotations

import functools
from typing import Any

import torch
import torch.distributed as dist
from opaque.api.engine import runtime
from opaque.api.engine.backend import KnownBackend
from opaque.api.engine.primitive import BackendProvider

_TORCH = BackendProvider(KnownBackend.TORCH)


@_TORCH.implements(runtime.distributed_is_initialized)
def distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


@_TORCH.implements(runtime.distributed_rank)
def distributed_rank() -> int:
    return dist.get_rank() if distributed_is_initialized() else 0


@_TORCH.implements(runtime.distributed_world_size)
def distributed_world_size() -> int:
    return dist.get_world_size() if distributed_is_initialized() else 1


@_TORCH.implements(runtime.distributed_all_reduce_)
def distributed_all_reduce_(value: torch.Tensor, op: str = "sum") -> None:
    operations = {
        "sum": dist.ReduceOp.SUM,
        "mean": dist.ReduceOp.AVG,
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
    if not distributed_is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialized. "
            "Call torch.distributed.init_process_group() first."
        )
    dist.all_reduce(value, op=reduce_op)


@_TORCH.implements(runtime.distributed_all_gather_object)
def distributed_all_gather_object(value: Any) -> list[Any]:
    if not distributed_is_initialized():
        return [value]
    gathered: list[Any] = [None] * distributed_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


@_TORCH.implements(runtime.distributed_reduce_scalar)
def distributed_reduce_scalar(
    value: float | int,
    op: str = "mean",
    device: Any = None,
    *,
    compute_dtype: Any = None,
) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"value must be a float or int, got {type(value)}")
    if compute_dtype is not None and not torch.is_floating_point(
        torch.empty((), dtype=compute_dtype)
    ):
        raise TypeError(
            f"compute_dtype must be a real floating-point dtype, got {compute_dtype!r}."
        )
    if isinstance(value, int) and compute_dtype is not None:
        raise TypeError("compute_dtype is only supported for floating-point values.")
    if not distributed_is_initialized():
        return float(value) if isinstance(value, int) and op == "mean" else value
    if device is None:
        backend = dist.get_backend()
        if backend == "nccl":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Distributed backend is 'nccl' but CUDA is not available; "
                    "provide an explicit `device` to `reduce_scalar` or initialize "
                    "with a CUDA-capable process."
                )
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            device = torch.device("cpu")
    is_integer = isinstance(value, int)
    tensor = torch.tensor(
        value,
        dtype=torch.int64 if is_integer else compute_dtype or torch.float32,
        device=device,
    )
    if is_integer and op == "mean":
        distributed_all_reduce_(tensor, op="sum")
        return tensor.item() / distributed_world_size()
    distributed_all_reduce_(tensor, op=op)
    return tensor.item()


@_TORCH.implements(runtime.distributed_barrier)
def distributed_barrier() -> None:
    if distributed_is_initialized():
        dist.barrier()


@_TORCH.implements(runtime.distributed_gather_for_metrics)
def distributed_gather_for_metrics(value: torch.Tensor) -> torch.Tensor:
    if not distributed_is_initialized():
        return value
    local = value.unsqueeze(0) if value.dim() == 0 else value
    gathered = [torch.empty_like(local) for _ in range(distributed_world_size())]
    dist.all_gather(gathered, local.contiguous())
    return torch.cat(gathered, dim=0)


@_TORCH.implements(runtime.distributed_dataset_subset)
def distributed_dataset_subset(dataset: Any, start: int, end: int) -> Any:
    from torch.utils.data import Subset

    return Subset(dataset, range(start, end))


@functools.cache
def _triton_importable() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


@functools.cache
def _probe_bf16(device_type: str) -> bool:
    try:
        if device_type == "cuda":
            return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if device_type == "mps":
            if not torch.backends.mps.is_available():
                return False
            probe = torch.ones(2, dtype=torch.bfloat16, device="mps")
            return (probe + probe).sum().item() == 4.0
        if device_type == "cpu":
            return True
    except Exception:
        return False
    return False


@_TORCH.implements(runtime.device_fused_kernels_available)
def device_fused_kernels_available() -> bool:
    return torch.cuda.is_available() and _triton_importable()


@_TORCH.implements(runtime.device_sdpa_autocast_under_vmap_broken)
@functools.cache
def device_sdpa_autocast_under_vmap_broken(device_type: str) -> bool:
    if device_type != "mps" or not torch.backends.mps.is_available():
        return False

    def _loss(scale, q, k, v):
        out = torch.nn.functional.scaled_dot_product_attention(q * scale, k, v)
        return out.float().sum()

    q = torch.randn(2, 1, 4, 8, device="mps")
    k = torch.randn(2, 1, 4, 8, device="mps")
    v = torch.randn(2, 1, 4, 8, device="mps", dtype=torch.bfloat16)
    scale = torch.tensor(1.0, device="mps")
    try:
        with torch.autocast(device_type="mps", dtype=torch.bfloat16):
            torch.vmap(torch.func.grad(_loss), in_dims=(None, 0, 0, 0))(scale, q, k, v)
    except RuntimeError:
        return True
    return False


@_TORCH.implements(runtime.device_capabilities)
def device_capabilities(device: Any) -> Any:
    from opaque.api.engine.device._capabilities import DeviceCapabilities

    device_type = torch.device(device).type
    supports_compile = device_type in ("cuda", "mps", "cpu")
    return DeviceCapabilities(
        device_type=device_type,
        supports_bf16=_probe_bf16(device_type),
        supports_compile=supports_compile,
        recommended_compile_backend="inductor" if supports_compile else None,
        supports_fused_kernels=(
            device_type == "cuda" and device_fused_kernels_available()
        ),
        peak_memory_trackable=device_type == "cuda",
        supports_pin_memory=device_type == "cuda",
    )


@_TORCH.implements(runtime.profiling_memory_stats)
def profiling_memory_stats(device: Any) -> Any:
    from opaque.api.engine.profiling._memory import MemoryStats

    device = torch.device(device)
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        peak = torch.cuda.max_memory_allocated(device) / (1024**3)
        total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        return MemoryStats(
            allocated_gb=allocated,
            reserved_gb=reserved,
            peak_gb=peak,
            free_gb=total - reserved,
            total_gb=total,
            exact_peak=True,
            exact_reserved=True,
            known_free=True,
            known_total=True,
        )
    if device.type == "mps":
        allocated = torch.mps.current_allocated_memory() / (1024**3)
        reserved = torch.mps.driver_allocated_memory() / (1024**3)
        total = torch.mps.recommended_max_memory() / (1024**3)
        return MemoryStats(
            allocated_gb=allocated,
            reserved_gb=reserved,
            peak_gb=reserved,
            free_gb=max(total - reserved, 0.0),
            total_gb=total,
            exact_peak=True,
            exact_reserved=True,
            known_free=True,
            known_total=True,
        )
    return MemoryStats()


@_TORCH.implements(runtime.profiling_reset_peak_memory)
def profiling_reset_peak_memory(device: Any) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "mps":
        torch.mps.empty_cache()


@_TORCH.implements(runtime.profiling_empty_cache)
def profiling_empty_cache(device: Any) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


@_TORCH.implements(runtime.profiling_synchronize)
def profiling_synchronize(device: Any) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


@_TORCH.implements(runtime.profiling_normalize_device)
def profiling_normalize_device(device: Any) -> torch.device:
    return torch.device(device)


@_TORCH.implements(runtime.profiling_trace_scope)
def profiling_trace_scope(label: str) -> Any:
    return torch.autograd.profiler.record_function(label)


@_TORCH.implements(runtime.functional_make_functional)
def functional_make_functional(
    mod: torch.nn.Module,
    disable_autograd_tracking: bool = False,
    partition_trainable: bool = False,
) -> Any:
    params_dict = dict(mod.named_parameters())
    if disable_autograd_tracking:
        params_dict = {name: param.detach() for name, param in params_dict.items()}
    if partition_trainable:
        original_params = dict(mod.named_parameters())
        trainable_params = {
            name: params_dict[name]
            for name, param in original_params.items()
            if param.requires_grad
        }
        frozen_params = {
            name: params_dict[name]
            for name, param in original_params.items()
            if not param.requires_grad
        }

        def fmodel_dict(params_dict_input: Any, *args: Any, **kwargs: Any) -> Any:
            return torch.func.functional_call(mod, params_dict_input, args, kwargs)

        return fmodel_dict, trainable_params, frozen_params

    params_names = list(params_dict)
    params_values = tuple(params_dict.values())

    def fmodel_tuple(new_params_values: Any, *args: Any, **kwargs: Any) -> Any:
        return torch.func.functional_call(
            mod,
            dict(zip(params_names, new_params_values, strict=True)),
            args,
            kwargs,
        )

    return fmodel_tuple, params_values


__all__: list[str] = []
