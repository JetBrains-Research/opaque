"""Optional provider capabilities used by engine runtime integrations.

These primitives deliberately sit outside the portable-core profile.  A
provider that only implements array/autodiff operations can therefore be
activated normally, while callers of a runtime integration receive the usual
``UnsupportedPrimitiveError`` at the public call site.
"""

from typing import Any

from opaque.api.engine.primitive import primitive


@primitive(name="opaque.runtime.distributed.is_initialized")
def distributed_is_initialized() -> bool:
    """Return whether distributed execution is initialized."""
    raise NotImplementedError


@primitive(name="opaque.runtime.distributed.rank")
def distributed_rank() -> int:
    """Return the current distributed rank."""
    raise NotImplementedError


@primitive(name="opaque.runtime.distributed.world_size")
def distributed_world_size() -> int:
    """Return the distributed world size."""
    raise NotImplementedError


@primitive(name="opaque.runtime.distributed.all_reduce_")
def distributed_all_reduce_(value: object, op: str = "sum") -> None:
    """Reduce ``value`` in place across distributed workers."""
    raise NotImplementedError


@primitive(name="opaque.runtime.distributed.all_gather_object")
def distributed_all_gather_object(value: Any) -> list[Any]:
    """Gather a Python value from every distributed worker."""
    raise NotImplementedError


@primitive(name="opaque.runtime.distributed.reduce_scalar")
def distributed_reduce_scalar(
    value: float | int,
    op: str = "mean",
    device: object = None,
    *,
    compute_dtype: object = None,
) -> float | int:
    """Reduce a Python scalar across distributed workers."""
    raise NotImplementedError


@primitive(name="opaque.runtime.distributed.barrier")
def distributed_barrier() -> None:
    """Synchronize all distributed workers."""
    raise NotImplementedError


@primitive(name="opaque.runtime.distributed.gather_for_metrics")
def distributed_gather_for_metrics(value: object) -> object:
    """Gather an array for distributed metric computation."""
    raise NotImplementedError


@primitive(name="opaque.runtime.distributed.dataset_subset")
def distributed_dataset_subset(dataset: Any, start: int, end: int) -> Any:
    """Create a provider-native dataset subset."""
    raise NotImplementedError


@primitive(name="opaque.runtime.device.capabilities")
def device_capabilities(device: object) -> Any:
    """Probe provider capabilities for ``device``."""
    raise NotImplementedError


@primitive(name="opaque.runtime.device.fused_kernels_available")
def device_fused_kernels_available() -> bool:
    """Return whether provider fused kernels are available."""
    raise NotImplementedError


@primitive(name="opaque.runtime.device.sdpa_autocast_under_vmap_broken")
def device_sdpa_autocast_under_vmap_broken(device_type: str) -> bool:
    """Probe the vectorized SDPA autocast compatibility of a device type."""
    raise NotImplementedError


@primitive(name="opaque.runtime.profiling.memory_stats")
def profiling_memory_stats(device: object) -> Any:
    """Read provider memory statistics for ``device``."""
    raise NotImplementedError


@primitive(name="opaque.runtime.profiling.reset_peak_memory")
def profiling_reset_peak_memory(device: object) -> None:
    """Reset provider peak-memory counters for ``device``."""
    raise NotImplementedError


@primitive(name="opaque.runtime.profiling.empty_cache")
def profiling_empty_cache(device: object) -> None:
    """Release unused provider memory for ``device``."""
    raise NotImplementedError


@primitive(name="opaque.runtime.profiling.synchronize")
def profiling_synchronize(device: object) -> None:
    """Synchronize queued work on ``device``."""
    raise NotImplementedError


@primitive(name="opaque.runtime.profiling.normalize_device")
def profiling_normalize_device(device: object) -> object:
    """Normalize a provider device specification."""
    raise NotImplementedError


@primitive(name="opaque.runtime.profiling.trace_scope")
def profiling_trace_scope(label: str) -> Any:
    """Create a provider-native profiling trace scope."""
    raise NotImplementedError


@primitive(name="opaque.runtime.functional.make_functional")
def functional_make_functional(
    mod: object,
    disable_autograd_tracking: bool = False,
    partition_trainable: bool = False,
) -> Any:
    """Convert a model to an explicit functional representation."""
    raise NotImplementedError


__all__ = [
    "device_capabilities",
    "device_fused_kernels_available",
    "device_sdpa_autocast_under_vmap_broken",
    "distributed_all_reduce_",
    "distributed_all_gather_object",
    "distributed_barrier",
    "distributed_dataset_subset",
    "distributed_gather_for_metrics",
    "distributed_is_initialized",
    "distributed_rank",
    "distributed_reduce_scalar",
    "distributed_world_size",
    "functional_make_functional",
    "profiling_empty_cache",
    "profiling_memory_stats",
    "profiling_normalize_device",
    "profiling_reset_peak_memory",
    "profiling_synchronize",
    "profiling_trace_scope",
]
