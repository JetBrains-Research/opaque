"""MLX runtime lifecycle, collective, and observability coverage."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from opaque import ops
from opaque.api.engine import runtime
from opaque.backend import clear_backend, set_backend
from opaque.mlx import distributed


class _Group:
    def __init__(self, rank: int = 1, size: int = 2) -> None:
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


@pytest.fixture(autouse=True)
def _clear_registered_group():
    distributed.clear_group()
    yield
    distributed.clear_group()


def test_backend_activation_and_runtime_probes_do_not_initialize_mlx(
    monkeypatch,
) -> None:
    def fail_initialize(*args, **kwargs):
        raise AssertionError("MLX distributed initialization must be explicit")

    monkeypatch.setattr(mx.distributed, "init", fail_initialize)
    clear_backend()
    set_backend("mlx")

    assert runtime.RuntimeProfile.DISTRIBUTED.supports()
    assert runtime.RuntimeProfile.OBSERVABILITY.supports()
    assert not runtime.distributed_initialized()
    assert runtime.distributed_rank() == 0
    assert runtime.distributed_world_size() == 1


def test_explicit_group_initialization_and_lifecycle(monkeypatch) -> None:
    group = _Group()
    calls = []

    def initialize(*, strict, backend, all_gather_factory=None):
        calls.append((strict, backend, all_gather_factory))
        return group

    monkeypatch.setattr(mx.distributed, "init", initialize)

    assert distributed.initialize(strict=True, backend="ring") is group
    assert calls == [(True, "ring", None)]
    assert runtime.distributed_initialized()
    assert runtime.distributed_rank() == 1
    assert runtime.distributed_world_size() == 2

    distributed.clear_group()
    assert not runtime.distributed_initialized()
    assert runtime.distributed_rank() == 0
    assert runtime.distributed_world_size() == 1


def test_registered_group_collectives_preserve_native_and_python_values(
    monkeypatch,
) -> None:
    group = _Group()
    distributed.register_group(group)

    monkeypatch.setattr(
        mx.distributed,
        "all_sum",
        lambda value, *, group: ops.multiply(value, group.size()),
    )
    monkeypatch.setattr(
        mx.distributed, "all_min", lambda value, *, group: ops.subtract(value, 1)
    )
    monkeypatch.setattr(
        mx.distributed, "all_max", lambda value, *, group: ops.add(value, 1)
    )
    monkeypatch.setattr(
        mx.distributed,
        "all_gather",
        lambda value, *, group: mx.concatenate((value, value), axis=0),
    )

    value = mx.array([1.0, 2.0], dtype=mx.float32)
    np.testing.assert_array_equal(
        ops.to_host(runtime.distributed_all_reduce(value)), np.array([2.0, 4.0])
    )
    np.testing.assert_array_equal(
        ops.to_host(runtime.distributed_all_reduce(value, runtime.ReduceOp.MEAN)),
        np.array([1.0, 2.0]),
    )
    np.testing.assert_array_equal(
        ops.to_host(runtime.distributed_all_reduce(value, runtime.ReduceOp.MIN)),
        np.array([0.0, 1.0]),
    )
    np.testing.assert_array_equal(
        ops.to_host(runtime.distributed_all_reduce(value, runtime.ReduceOp.MAX)),
        np.array([2.0, 3.0]),
    )
    np.testing.assert_array_equal(
        ops.to_host(runtime.distributed_all_reduce(value, runtime.ReduceOp.PRODUCT)),
        np.array([1.0, 4.0]),
    )
    np.testing.assert_array_equal(
        ops.to_host(runtime.distributed_all_gather(value, axis=0)),
        np.array([1.0, 2.0, 1.0, 2.0]),
    )

    large_integer = 2**60 + 3
    assert runtime.distributed_all_reduce(large_integer) == large_integer * 2
    assert runtime.distributed_all_gather_object({"rank": 1}) == [
        {"rank": 1},
        {"rank": 1},
    ]
    runtime.distributed_barrier("test-runtime")


def test_runtime_observability_reports_only_available_mlx_measurements(
    monkeypatch,
) -> None:
    synchronized = []
    cleared = []
    monkeypatch.setattr(
        mx, "synchronize", lambda device=None: synchronized.append(device)
    )
    monkeypatch.setattr(mx, "clear_cache", lambda: cleared.append(True))
    monkeypatch.setattr(mx, "get_active_memory", lambda: 11)
    monkeypatch.setattr(mx, "get_peak_memory", lambda: 17)

    runtime.synchronize()
    runtime.clear_memory_cache()
    stats = runtime.memory_stats()

    assert synchronized == [None]
    assert cleared == [True]
    assert stats == runtime.MemoryStats(
        active_bytes=11,
        cached_bytes=None,
        peak_active_bytes=17,
        capacity_bytes=None,
    )


def test_two_rank_ring_collectives_preserve_array_and_scalar_contracts() -> None:
    launcher = shutil.which("mlx.launch")
    assert launcher is not None
    worker = Path(__file__).with_name("_two_rank_worker.py")
    result = subprocess.run(
        [
            launcher,
            "--backend",
            "ring",
            "-n",
            "2",
            sys.executable,
            str(worker),
        ],
        check=True,
        capture_output=True,
        cwd=Path(__file__).parents[4],
        text=True,
        timeout=90,
    )
    records = [
        json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")
    ]

    assert sorted(records, key=lambda record: record["rank"]) == [
        {
            "rank": 0,
            "world_size": 2,
            "reduced": [3.0],
            "scalar": (2**61) + 1,
            "gathered": [[0, 10, 1, 11]],
            "objects": [{"rank": 0}, {"rank": 1}],
            "dpsgd_total": [3.0],
            "dpsgd_step": 1,
            "dpftrl_total": [3.0],
            "dpftrl_step": 1,
        },
        {
            "rank": 1,
            "world_size": 2,
            "reduced": [3.0],
            "scalar": (2**61) + 1,
            "gathered": [[0, 10, 1, 11]],
            "objects": [{"rank": 0}, {"rank": 1}],
            "dpsgd_total": [3.0],
            "dpsgd_step": 1,
            "dpftrl_total": [3.0],
            "dpftrl_step": 1,
        },
    ]
