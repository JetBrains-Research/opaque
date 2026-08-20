"""Behavior tests for dynamically discovered CI package matrices."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parents[2] / ".github" / "scripts"


def _load_discovery_module():
    script_dir = str(_SCRIPT_DIR)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    return importlib.import_module("discover_package_matrices")


def _paths(shards: list[dict[str, object]]) -> list[str]:
    paths: list[str] = []
    for shard in shards:
        shard_paths = shard["paths"]
        assert isinstance(shard_paths, list)
        assert all(isinstance(path, str) for path in shard_paths)
        paths.extend(shard_paths)
    return paths


def test_discovered_test_matrices_cover_each_package_once() -> None:
    outputs = _load_discovery_module()._outputs()
    expected_paths = {
        package["path"]
        for package in (
            *outputs["python_build_packages"],
            *outputs["native_build_packages"],
        )
        if package["path"] != "."
    }

    test_paths = _paths(outputs["test_shards"])
    assert test_paths.count("tests") == 1
    assert set(test_paths) - {"tests"} == expected_paths
    assert len(test_paths) == len(set(test_paths))

    for matrix_name in ("cuda_test_shards", "distributed_test_shards"):
        matrix_paths = _paths(outputs[matrix_name])
        assert set(matrix_paths) - {"tests"} <= expected_paths
        assert len(matrix_paths) == len(set(matrix_paths))
