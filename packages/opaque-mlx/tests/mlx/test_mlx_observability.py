"""Observability runtime coverage for the MLX provider."""

from __future__ import annotations

import pytest

from opaque.api.engine import runtime
from opaque.mlx import mlx_backend

mx = pytest.importorskip("mlx.core")


def test_observability_capabilities_are_truthful() -> None:
    backend = mlx_backend()

    assert runtime.RuntimeProfile.OBSERVABILITY.supports(backend)
    assert runtime.synchronize.supports(backend)
    assert runtime.memory_stats.supports(backend)
    assert runtime.clear_memory_cache.supports(backend)
    assert runtime.reset_peak_memory.supports(backend)
    assert not runtime.trace_scope.supports(backend)


def test_observability_operations_map_to_mlx_apis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opaque.api.mlx.backend import _runtime as provider

    calls: list[tuple[str, object | None]] = []
    device = object()
    monkeypatch.setattr(
        provider.mx,
        "synchronize",
        lambda selected=None: calls.append(("synchronize", selected)),
    )
    monkeypatch.setattr(
        provider.mx,
        "clear_cache",
        lambda: calls.append(("clear_cache", None)),
    )
    monkeypatch.setattr(
        provider.mx,
        "reset_peak_memory",
        lambda: calls.append(("reset_peak_memory", None)),
    )

    assert provider.synchronize(device) is None
    assert provider.clear_memory_cache() is None
    assert provider.reset_peak_memory() is None
    assert calls == [
        ("synchronize", device),
        ("clear_cache", None),
        ("reset_peak_memory", None),
    ]


def test_memory_stats_maps_mlx_allocator_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opaque.api.mlx.backend import _runtime as provider

    monkeypatch.setattr(provider.mx, "get_active_memory", lambda: 128)
    monkeypatch.setattr(provider.mx, "get_cache_memory", lambda: 64)
    monkeypatch.setattr(provider.mx, "get_peak_memory", lambda: 256)

    assert provider.memory_stats() == runtime.MemoryStats(
        active_bytes=128,
        cached_bytes=64,
        peak_active_bytes=256,
        capacity_bytes=None,
    )
