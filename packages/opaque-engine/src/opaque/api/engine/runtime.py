"""Optional provider capabilities used by engine runtime integrations.

These primitives deliberately sit outside the portable-core profile.  A
provider that only implements array/autodiff operations can therefore be
activated normally, while callers of a runtime integration receive the usual
``UnsupportedPrimitiveError`` at the public call site.
"""

from opaque.api.engine.primitive import Primitive


def _runtime(name: str) -> Primitive:
    return Primitive(f"opaque.runtime.{name}")


distributed_is_initialized = _runtime("distributed.is_initialized")
distributed_rank = _runtime("distributed.rank")
distributed_world_size = _runtime("distributed.world_size")
distributed_all_reduce_ = _runtime("distributed.all_reduce_")
distributed_all_gather_object = _runtime("distributed.all_gather_object")
distributed_reduce_scalar = _runtime("distributed.reduce_scalar")
distributed_barrier = _runtime("distributed.barrier")
distributed_gather_for_metrics = _runtime("distributed.gather_for_metrics")
distributed_dataset_subset = _runtime("distributed.dataset_subset")

device_capabilities = _runtime("device.capabilities")
device_fused_kernels_available = _runtime("device.fused_kernels_available")
device_sdpa_autocast_under_vmap_broken = _runtime(
    "device.sdpa_autocast_under_vmap_broken"
)

profiling_memory_stats = _runtime("profiling.memory_stats")
profiling_reset_peak_memory = _runtime("profiling.reset_peak_memory")
profiling_empty_cache = _runtime("profiling.empty_cache")
profiling_synchronize = _runtime("profiling.synchronize")
profiling_normalize_device = _runtime("profiling.normalize_device")
profiling_trace_scope = _runtime("profiling.trace_scope")

functional_make_functional = _runtime("functional.make_functional")

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
