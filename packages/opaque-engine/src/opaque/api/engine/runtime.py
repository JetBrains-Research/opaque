"""Provider-neutral optional runtime capabilities.

The named profiles are discovery contracts, not activation requirements. A
provider may implement only the portable compute core, while callers can use
:func:`supports_profile` or individual primitive support checks before using
distributed, observability, or allocator-specific integrations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from opaque.api.engine.primitive import Primitive, primitive


class ReduceOp(StrEnum):
    """Portable distributed reduction operations."""

    SUM = "sum"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    PRODUCT = "product"


@dataclass(frozen=True)
class MemoryStats:
    """Normalized provider memory observations in bytes.

    A field is ``None`` when the provider or selected device cannot report the
    corresponding observation. Providers must not replace unavailable values
    with fabricated zeros.

    ``active_bytes`` is memory currently occupied by live arrays,
    ``cached_bytes`` is unused memory retained by the allocator for reuse,
    ``peak_active_bytes`` is the observed active-memory high-water mark, and
    ``capacity_bytes`` is the device's total usable memory when exposed.
    """

    active_bytes: int | None = None
    cached_bytes: int | None = None
    peak_active_bytes: int | None = None
    capacity_bytes: int | None = None


@primitive(name="opaque.runtime.distributed.initialized")
def distributed_initialized() -> bool:
    """Return whether a distributed process group is live.

    True for any initialized group, including a single-rank one — the
    distinction matters to callers that must issue collectives on every
    rank of a live group regardless of its size.
    """
    raise NotImplementedError


@primitive(name="opaque.runtime.distributed.rank")
def distributed_rank() -> int:
    """Return the current distributed rank."""
    raise NotImplementedError


@primitive(name="opaque.runtime.distributed.world_size")
def distributed_world_size() -> int:
    """Return the distributed world size."""
    raise NotImplementedError


@primitive(name="opaque.runtime.distributed.all_reduce")
def distributed_all_reduce(value: object, op: ReduceOp = ReduceOp.SUM) -> object:
    """Return ``value`` reduced across distributed workers.

    Accepts a native array or a Python ``float`` / ``int`` and returns the same
    kind.  An array reduces in its own dtype.  A Python scalar reduces at the
    provider's widest exact representation for that kind — ``float`` is a
    float64 and must not be narrowed to reach the wire — so the result never
    depends on a framework-global default dtype.  ``int`` stays exact; an
    integer ``mean`` divides in Python rather than through a float tensor.

    Args:
        value: Native array, or a Python ``float`` / ``int``. ``bool`` is
            rejected.
        op: ``sum``, ``mean``, ``max``, ``min``, or ``product``.

    Returns:
        The reduced value, of the same kind as ``value``. Outside a live
        process group, providers return it unchanged.
    """
    raise NotImplementedError


@primitive(name="opaque.runtime.distributed.all_gather")
def distributed_all_gather(value: object, *, axis: int = 0) -> object:
    """Gather native arrays and concatenate rank-local values along ``axis``."""
    raise NotImplementedError


@primitive(name="opaque.runtime.distributed.all_gather_object")
def distributed_all_gather_object(value: object) -> list[object]:
    """Gather a Python value from every distributed worker."""
    raise NotImplementedError


@primitive(name="opaque.runtime.distributed.barrier")
def distributed_barrier(name: str | None = None) -> None:
    """Synchronize all distributed workers at an optional named barrier."""
    raise NotImplementedError


@primitive(name="opaque.runtime.observability.synchronize")
def synchronize(device: object | None = None) -> None:
    """Wait for queued provider work on ``device`` to complete."""
    raise NotImplementedError


@primitive(name="opaque.runtime.observability.memory_stats")
def memory_stats(device: object | None = None) -> MemoryStats:
    """Return normalized memory observations for ``device``."""
    raise NotImplementedError


@primitive(name="opaque.runtime.observability.clear_memory_cache")
def clear_memory_cache(device: object | None = None) -> None:
    """Release unused provider allocator memory for ``device``."""
    raise NotImplementedError


@primitive(name="opaque.runtime.observability.reset_peak_memory")
def reset_peak_memory(device: object | None = None) -> None:
    """Reset provider peak-active-memory counters for ``device``."""
    raise NotImplementedError


@primitive(name="opaque.runtime.observability.peak_memory_trackable")
def peak_memory_trackable(device: object | None = None) -> bool:
    """Whether ``device`` exposes a cheap per-step peak-memory counter.

    ``True`` means :func:`reset_peak_memory` is a counter reset that is safe
    to run every step (e.g. CUDA). ``False`` means resetting is heavyweight
    (e.g. MPS, where the reset releases cached allocator blocks) and callers
    should reset between measured configurations only. Deliberately not part
    of the observability profile: providers without the probe keep working,
    and per-step reset is simply skipped.
    """
    raise NotImplementedError


@primitive(name="opaque.runtime.observability.trace_scope")
def trace_scope(label: str) -> object:
    """Return a provider-native trace annotation context manager."""
    raise NotImplementedError


class RuntimeProfile(StrEnum):
    """Named optional runtime integration profiles."""

    DISTRIBUTED = "distributed"
    OBSERVABILITY = "observability"

    @property
    def primitives(self) -> tuple[Primitive, ...]:
        """Return the primitive declarations required by this profile."""
        return profile_primitives(self)

    def supports(self, backend: object | str | None = None) -> bool:
        """Return whether ``backend`` implements this complete profile."""
        return supports_profile(self, backend)


RUNTIME_PROFILE_VERSION = 2
"""Version of the named optional runtime profile contract."""


_RUNTIME_PROFILES: dict[RuntimeProfile, tuple[Primitive, ...]] = {
    RuntimeProfile.DISTRIBUTED: (
        distributed_initialized,
        distributed_all_reduce,
        distributed_all_gather,
        distributed_all_gather_object,
        distributed_rank,
        distributed_world_size,
        distributed_barrier,
    ),
    RuntimeProfile.OBSERVABILITY: (
        synchronize,
        memory_stats,
    ),
}


def profile_primitives(
    profile: RuntimeProfile | str,
) -> tuple[Primitive, ...]:
    """Return the declarations required by a named runtime profile."""
    return _RUNTIME_PROFILES[RuntimeProfile(profile)]


def supports_profile(
    profile: RuntimeProfile | str,
    backend: object | str | None = None,
) -> bool:
    """Return whether ``backend`` registered every primitive in ``profile``."""
    return all(operation.supports(backend) for operation in profile_primitives(profile))


__all__ = [
    "MemoryStats",
    "RUNTIME_PROFILE_VERSION",
    "ReduceOp",
    "RuntimeProfile",
    "clear_memory_cache",
    "distributed_all_gather",
    "distributed_all_gather_object",
    "distributed_all_reduce",
    "distributed_barrier",
    "distributed_initialized",
    "distributed_rank",
    "distributed_world_size",
    "memory_stats",
    "profile_primitives",
    "reset_peak_memory",
    "supports_profile",
    "synchronize",
    "trace_scope",
]
