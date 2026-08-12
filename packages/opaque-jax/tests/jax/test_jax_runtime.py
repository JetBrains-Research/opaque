"""Distributed runtime coverage for the JAX provider."""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
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
from opaque.jax import jax_backend

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


@pytest.fixture(autouse=True)
def _reset_backend() -> None:
    clear_backend()
    yield
    clear_backend()


def test_singleton_public_distributed_profile() -> None:
    backend = jax_backend()
    value = jnp.arange(6, dtype=jnp.float32).reshape(3, 2)

    with use_backend(backend):
        assert runtime.supports_profile(runtime.RuntimeProfile.DISTRIBUTED)
        assert get_rank() == 0
        assert get_world_size() == 1
        assert not is_distributed()
        assert wait_for_everyone() is None

        reduced = all_reduce(value, op="sum")
        gathered = gather_for_metrics(value)
        objects = runtime.distributed_all_gather_object({"rank": 0})

    np.testing.assert_array_equal(reduced, value)
    np.testing.assert_array_equal(gathered, value)
    assert isinstance(reduced, jax.Array)
    assert isinstance(gathered, jax.Array)
    assert reduced is not value
    assert gathered is not value
    assert objects == [{"rank": 0}]


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (runtime.ReduceOp.SUM, [4.0, 6.0]),
        (runtime.ReduceOp.MEAN, [2.0, 3.0]),
        (runtime.ReduceOp.MIN, [1.0, 2.0]),
        (runtime.ReduceOp.MAX, [3.0, 4.0]),
        (runtime.ReduceOp.PRODUCT, [3.0, 8.0]),
    ],
)
def test_reduction_maps_process_gather_to_local_jax_op(
    monkeypatch: pytest.MonkeyPatch,
    op: runtime.ReduceOp,
    expected: list[float],
) -> None:
    from opaque.api.jax.backend import _runtime as provider

    calls: list[tuple[Any, bool]] = []
    monkeypatch.setattr(provider.jax, "process_count", lambda: 2)

    def process_allgather(value: Any, tiled: bool = False) -> np.ndarray:
        calls.append((value, tiled))
        return np.asarray([[1.0, 2.0], [3.0, 4.0]])

    monkeypatch.setattr(
        provider.multihost_utils, "process_allgather", process_allgather
    )

    result = provider.distributed_all_reduce(jnp.array([1.0, 2.0]), op)

    np.testing.assert_array_equal(result, expected)
    assert isinstance(result, jax.Array)
    assert len(calls) == 1
    assert calls[0][1] is False


def test_ragged_nonzero_axis_gather_pads_and_trims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opaque.api.jax.backend import _runtime as provider

    value = jnp.array([[1.0], [2.0]])
    metadata = [
        ("float32", 2, (2, 1)),
        ("float32", 2, (2, 2)),
    ]
    monkeypatch.setattr(provider.jax, "process_count", lambda: 2)
    monkeypatch.setattr(
        provider, "distributed_all_gather_object", lambda value: metadata
    )

    def process_allgather(value: Any, tiled: bool = False) -> np.ndarray:
        assert np.asarray(value).shape == (2, 2)
        assert tiled is False
        return np.asarray(
            [
                [[1.0, 0.0], [2.0, 0.0]],
                [[3.0, 4.0], [5.0, 6.0]],
            ]
        )

    monkeypatch.setattr(
        provider.multihost_utils, "process_allgather", process_allgather
    )

    result = provider.distributed_all_gather(value, axis=1)

    np.testing.assert_array_equal(result, [[1.0, 3.0, 4.0], [2.0, 5.0, 6.0]])
    assert isinstance(result, jax.Array)


def test_python_integer_reduction_preserves_host_integer_exactness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opaque.api.jax.backend import _runtime as provider

    base = 2**40
    monkeypatch.setattr(provider.jax, "process_count", lambda: 2)

    def process_allgather(value: Any, tiled: bool = False) -> np.ndarray:
        assert np.asarray(value).dtype == np.dtype(np.int64)
        assert tiled is False
        return np.asarray([base, base + 1], dtype=np.int64)

    monkeypatch.setattr(
        provider.multihost_utils, "process_allgather", process_allgather
    )

    assert provider.distributed_all_reduce(base, runtime.ReduceOp.SUM) == 2 * base + 1


def test_object_gather_uses_length_prefixed_process_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opaque.api.jax.backend import _runtime as provider

    local = {"rank": 0, "payload": b"short"}
    remote = {"rank": 1, "payload": b"a longer value"}
    local_bytes = pickle.dumps(local, protocol=pickle.HIGHEST_PROTOCOL)
    remote_bytes = pickle.dumps(remote, protocol=pickle.HIGHEST_PROTOCOL)
    width = max(len(local_bytes), len(remote_bytes))
    monkeypatch.setattr(provider.jax, "process_count", lambda: 2)

    def padded(payload: bytes) -> np.ndarray:
        result = np.zeros(width, dtype=np.uint8)
        result[: len(payload)] = np.frombuffer(payload, dtype=np.uint8)
        return result

    def process_allgather(value: Any, tiled: bool = False) -> np.ndarray:
        assert tiled is False
        if np.asarray(value).shape == (1,):
            return np.asarray([[len(local_bytes)], [len(remote_bytes)]])
        return np.stack([padded(local_bytes), padded(remote_bytes)])

    monkeypatch.setattr(
        provider.multihost_utils, "process_allgather", process_allgather
    )

    assert provider.distributed_all_gather_object(local) == [local, remote]


def test_rank_and_named_barrier_map_to_jax_process_apis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opaque.api.jax.backend import _runtime as provider

    names: list[str] = []
    monkeypatch.setattr(provider.jax, "process_index", lambda: 3)
    monkeypatch.setattr(provider.jax, "process_count", lambda: 4)
    monkeypatch.setattr(provider.multihost_utils, "sync_global_devices", names.append)

    assert provider.distributed_rank() == 3
    assert provider.distributed_world_size() == 4
    assert provider.distributed_barrier("checkpoint") is None
    assert names == ["checkpoint"]


def test_distributed_runtime_rejects_invalid_values_and_axes() -> None:
    from opaque.api.jax.backend import _runtime as provider

    with pytest.raises(TypeError, match="array, float, or int"):
        provider.distributed_all_reduce(True)
    with pytest.raises(ValueError, match="Invalid reduction"):
        provider.distributed_all_reduce(1, "median")
    with pytest.raises(TypeError, match="JAX array"):
        provider.distributed_all_gather([1, 2])
    with pytest.raises(IndexError, match="out of bounds"):
        provider.distributed_all_gather(jnp.ones((2, 2)), axis=2)
