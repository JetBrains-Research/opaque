"""Observability runtime coverage for the JAX provider."""

from __future__ import annotations

from typing import Any

import pytest

from opaque.api.engine import runtime
from opaque.jax import jax_backend

jax = pytest.importorskip("jax")


def test_observability_capabilities_are_truthful() -> None:
    backend = jax_backend()

    assert runtime.RuntimeProfile.OBSERVABILITY.supports(backend)
    assert runtime.synchronize.supports(backend)
    assert runtime.memory_stats.supports(backend)
    assert runtime.trace_scope.supports(backend)
    assert not runtime.clear_memory_cache.supports(backend)
    assert not runtime.reset_peak_memory.supports(backend)


def test_synchronize_and_trace_scope_map_to_jax_apis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opaque.api.jax.backend import _runtime as provider

    barrier_calls: list[None] = []
    marker = object()
    labels: list[str] = []
    monkeypatch.setattr(
        provider.jax,
        "effects_barrier",
        lambda: barrier_calls.append(None),
    )
    monkeypatch.setattr(
        provider.jax.profiler,
        "TraceAnnotation",
        lambda label: labels.append(label) or marker,
    )

    assert provider.synchronize() is None
    assert provider.trace_scope("opaque::test") is marker
    assert barrier_calls == [None]
    assert labels == ["opaque::test"]


def test_memory_stats_maps_available_nonnegative_device_fields() -> None:
    from opaque.api.jax.backend import _runtime as provider

    class Device:
        def memory_stats(self) -> dict[str, int]:
            return {
                "bytes_in_use": 128,
                "pool_bytes": 512,
                "peak_bytes_in_use": 256,
                "bytes_limit": 1024,
            }

    assert provider.memory_stats(Device()) == runtime.MemoryStats(
        active_bytes=128,
        cached_bytes=384,
        peak_active_bytes=256,
        capacity_bytes=1024,
    )


@pytest.mark.parametrize("native_stats", [None, {"bytes_in_use": -1}])
def test_memory_stats_preserves_unavailable_values(
    native_stats: dict[str, int] | None,
) -> None:
    from opaque.api.jax.backend import _runtime as provider

    class Device:
        def memory_stats(self) -> Any:
            return native_stats

    assert provider.memory_stats(Device()) == runtime.MemoryStats()
