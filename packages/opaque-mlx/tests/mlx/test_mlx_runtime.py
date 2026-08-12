"""Distributed runtime coverage for the MLX provider."""

from __future__ import annotations

import pickle
from typing import Any

import pytest

from opaque.api.engine import runtime
from opaque.api.engine.backend import clear_backend, use_backend
from opaque.distributed import (
    all_reduce,
    gather_for_metrics,
    get_rank,
    get_world_size,
    is_distributed,
    wait_for_everyone,
)
from opaque.mlx import mlx_backend

mx = pytest.importorskip("mlx.core")


@pytest.fixture(autouse=True)
def _reset_backend() -> None:
    clear_backend()
    yield
    clear_backend()


def _values(value: Any) -> Any:
    mx.eval(value)
    return value.tolist()


class _Group:
    def __init__(self, rank: int = 0, size: int = 2) -> None:
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def test_singleton_public_distributed_profile() -> None:
    backend = mlx_backend()
    value = mx.arange(6, dtype=mx.float32).reshape(3, 2)

    with use_backend(backend):
        assert runtime.supports_profile(runtime.RuntimeProfile.DISTRIBUTED)
        assert get_rank() == 0
        assert get_world_size() == 1
        assert not is_distributed()
        assert wait_for_everyone() is None

        reduced = all_reduce(value, op="sum")
        gathered = gather_for_metrics(value)
        objects = runtime.distributed_all_gather_object({"rank": 0})

    assert _values(reduced) == _values(value)
    assert _values(gathered) == _values(value)
    assert isinstance(reduced, mx.array)
    assert isinstance(gathered, mx.array)
    assert reduced is not value
    assert gathered is not value
    assert objects == [{"rank": 0}]


@pytest.mark.parametrize(
    ("op", "native_name", "native_result", "expected"),
    [
        (runtime.ReduceOp.SUM, "all_sum", [4.0, 6.0], [4.0, 6.0]),
        (runtime.ReduceOp.MEAN, "all_sum", [4.0, 6.0], [2.0, 3.0]),
        (runtime.ReduceOp.MIN, "all_min", [1.0, 2.0], [1.0, 2.0]),
        (runtime.ReduceOp.MAX, "all_max", [3.0, 4.0], [3.0, 4.0]),
    ],
)
def test_reduction_maps_to_native_mlx_collective(
    monkeypatch: pytest.MonkeyPatch,
    op: runtime.ReduceOp,
    native_name: str,
    native_result: list[float],
    expected: list[float],
) -> None:
    from opaque.api.mlx.backend import _runtime as provider

    group = _Group()
    calls: list[tuple[Any, Any]] = []
    monkeypatch.setattr(provider, "_global_group", lambda: group)

    def collective(value: Any, *, group: Any) -> Any:
        calls.append((value, group))
        return mx.array(native_result)

    monkeypatch.setattr(provider.mx.distributed, native_name, collective)

    result = provider.distributed_all_reduce(mx.array([1.0, 2.0]), op)

    assert _values(result) == expected
    assert calls == [(calls[0][0], group)]


def test_product_reduction_falls_back_to_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opaque.api.mlx.backend import _runtime as provider

    group = _Group()
    monkeypatch.setattr(provider, "_global_group", lambda: group)

    def all_gather(value: Any, *, group: Any) -> Any:
        assert group is not None
        assert tuple(value.shape) == (1, 2)
        return mx.array([[2.0, 3.0], [4.0, 5.0]])

    monkeypatch.setattr(provider.mx.distributed, "all_gather", all_gather)

    result = provider.distributed_all_reduce(
        mx.array([2.0, 3.0]), runtime.ReduceOp.PRODUCT
    )

    assert _values(result) == [8.0, 15.0]


def test_python_integer_reduction_uses_int64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opaque.api.mlx.backend import _runtime as provider

    group = _Group()
    base = 2**40
    monkeypatch.setattr(provider, "_global_group", lambda: group)

    def all_sum(value: Any, *, group: Any) -> Any:
        assert value.dtype == mx.int64
        return mx.array(2 * base + 1, dtype=mx.int64)

    monkeypatch.setattr(provider.mx.distributed, "all_sum", all_sum)

    assert provider.distributed_all_reduce(base, runtime.ReduceOp.SUM) == 2 * base + 1


def test_ragged_nonzero_axis_gather_pads_and_trims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opaque.api.mlx.backend import _runtime as provider

    group = _Group()
    value = mx.array([[1.0], [2.0]])
    metadata = [
        ("mlx.core.float32", 2, (2, 1)),
        ("mlx.core.float32", 2, (2, 2)),
    ]
    monkeypatch.setattr(provider, "_global_group", lambda: group)
    monkeypatch.setattr(
        provider, "distributed_all_gather_object", lambda value: metadata
    )

    def all_gather(value: Any, *, group: Any) -> Any:
        assert group is not None
        assert tuple(value.shape) == (2, 2)
        return mx.array(
            [
                [1.0, 2.0],
                [0.0, 0.0],
                [3.0, 5.0],
                [4.0, 6.0],
            ]
        )

    monkeypatch.setattr(provider.mx.distributed, "all_gather", all_gather)

    result = provider.distributed_all_gather(value, axis=1)

    assert _values(result) == [[1.0, 3.0, 4.0], [2.0, 5.0, 6.0]]
    assert isinstance(result, mx.array)


def test_object_gather_uses_length_prefixed_native_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opaque.api.mlx.backend import _runtime as provider

    group = _Group()
    local = {"rank": 0, "payload": b"short"}
    remote = {"rank": 1, "payload": b"a longer value"}
    local_bytes = pickle.dumps(local, protocol=pickle.HIGHEST_PROTOCOL)
    remote_bytes = pickle.dumps(remote, protocol=pickle.HIGHEST_PROTOCOL)
    width = max(len(local_bytes), len(remote_bytes))
    monkeypatch.setattr(provider, "_global_group", lambda: group)

    def padded(payload: bytes) -> list[int]:
        return [*payload, *([0] * (width - len(payload)))]

    def all_gather(value: Any, *, group: Any) -> Any:
        assert group is not None
        if tuple(value.shape) == (1,):
            return mx.array([len(local_bytes), len(remote_bytes)], dtype=mx.uint32)
        return mx.array(padded(local_bytes) + padded(remote_bytes), dtype=mx.uint8)

    monkeypatch.setattr(provider.mx.distributed, "all_gather", all_gather)

    assert provider.distributed_all_gather_object(local) == [local, remote]


def test_cached_group_rank_and_size_map_to_mlx_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opaque.api.mlx.backend import _runtime as provider

    group = _Group(rank=3, size=4)
    calls: list[bool] = []
    provider._global_group.cache_clear()

    def init(*, strict: bool) -> _Group:
        calls.append(strict)
        return group

    monkeypatch.setattr(provider.mx.distributed, "init", init)

    assert provider.distributed_rank() == 3
    assert provider.distributed_world_size() == 4
    assert provider.distributed_rank() == 3
    assert calls == [False]
    provider._global_group.cache_clear()


def test_distributed_runtime_rejects_invalid_values_and_axes() -> None:
    from opaque.api.mlx.backend import _runtime as provider

    with pytest.raises(TypeError, match="array, float, or int"):
        provider.distributed_all_reduce(True)
    with pytest.raises(ValueError, match="Invalid reduction"):
        provider.distributed_all_reduce(1, "median")
    with pytest.raises(TypeError, match="MLX array"):
        provider.distributed_all_gather([1, 2])
    with pytest.raises(IndexError, match="out of bounds"):
        provider.distributed_all_gather(mx.ones((2, 2)), axis=2)
