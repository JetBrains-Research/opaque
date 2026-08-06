from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
from benchmarks.core import (
    SCHEMA_VERSION,
    BenchmarkCase,
    CaseRun,
    _parse_dirty_paths,
    benchmark_source_paths,
    capture_sources,
    source_digest,
    summarize_samples,
    validate_result,
)

if TYPE_CHECKING:
    from pathlib import Path


def _valid_result(root: Path) -> dict[str, object]:
    source = root / "implementation.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    sources = capture_sources(root, ["implementation.py"])
    return {
        "schema_version": SCHEMA_VERSION,
        "case": {
            "id": "example.case",
            "description": "Example benchmark.",
            "config": {"size": 8},
            "sources": sources,
            "source_digest": source_digest(sources),
        },
        "provenance": {
            "timestamp_utc": "2026-08-06T05:00:00Z",
            "command": ["python", "-m", "benchmarks", "run", "example.case"],
            "git": {"commit": "a" * 40, "dirty": False, "dirty_paths": []},
            "hardware": {
                "machine": "arm64",
                "cpu": "Example CPU",
                "memory_bytes": 1024,
                "accelerator": None,
            },
            "software": {
                "platform": "Example OS",
                "python": "3.12.0",
                "packages": {"opaque": "0.1.0", "torch": "2.10.0"},
            },
        },
        "measurements": [
            {
                "name": "baseline",
                "parameters": {"size": 8},
                "metrics": {
                    "wall_time_ms": {
                        "value": 1.5,
                        "unit": "ms",
                        "statistic": "median",
                        "samples": [1.0, 1.5, 2.0],
                    }
                },
            }
        ],
    }


def test_summarize_samples_reports_robust_statistics() -> None:
    summary = summarize_samples([4.0, 1.0, 3.0, 2.0])

    assert summary == {
        "count": 4,
        "min": 1.0,
        "median": 2.5,
        "mean": 2.5,
        "max": 4.0,
        "stdev": pytest.approx(math.sqrt(5 / 3)),
    }


def test_parse_dirty_paths_preserves_first_character_and_rename_target() -> None:
    status = " M Cargo.lock\n?? benchmarks/new.py\nR  old.py -> new.py\n"

    assert _parse_dirty_paths(status) == [
        "Cargo.lock",
        "benchmarks/new.py",
        "new.py",
    ]


def test_capture_sources_expands_directories(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.py").write_text("ONE = 1\n", encoding="utf-8")
    (source / "two.rs").write_text("const TWO: u8 = 2;\n", encoding="utf-8")

    assert set(capture_sources(tmp_path, ["source"])) == {
        "source/one.py",
        "source/two.rs",
    }


def test_benchmark_sources_always_include_dependency_locks() -> None:
    case = BenchmarkCase(
        case_id="example",
        description="Example.",
        source_files=("implementation.py",),
        presets={"smoke": {}},
        devices=("cpu",),
        runner=lambda config, options: CaseRun([]),
    )

    assert benchmark_source_paths(case) == (
        "implementation.py",
        "pyproject.toml",
        "uv.lock",
    )


@pytest.mark.parametrize("samples", [[], [1.0, float("nan")], [float("inf")]])
def test_summarize_samples_rejects_unusable_input(samples: list[float]) -> None:
    with pytest.raises(ValueError, match="finite value"):
        summarize_samples(samples)


def test_validate_result_accepts_complete_fresh_result(tmp_path: Path) -> None:
    assert validate_result(_valid_result(tmp_path), root=tmp_path) == []


def test_validate_result_rejects_missing_provenance(tmp_path: Path) -> None:
    result = _valid_result(tmp_path)
    del result["provenance"]["hardware"]["cpu"]  # type: ignore[index]

    errors = validate_result(result, root=tmp_path)

    assert any("provenance.hardware.cpu" in error for error in errors)


def test_validate_result_detects_changed_benchmark_source(tmp_path: Path) -> None:
    result = _valid_result(tmp_path)
    (tmp_path / "implementation.py").write_text("VALUE = 2\n", encoding="utf-8")

    errors = validate_result(result, root=tmp_path)

    assert any("stale source implementation.py" in error for error in errors)


def test_validate_result_rejects_non_finite_metric(tmp_path: Path) -> None:
    result = _valid_result(tmp_path)
    metric = result["measurements"][0]["metrics"]["wall_time_ms"]  # type: ignore[index]
    metric["value"] = float("nan")

    errors = validate_result(result, root=tmp_path)

    assert any("must be finite" in error for error in errors)


def test_validate_result_rejects_non_finite_comparison(tmp_path: Path) -> None:
    result = _valid_result(tmp_path)
    result["comparisons"] = [
        {
            "name": "speedup",
            "baseline": "baseline",
            "candidate": "candidate",
            "metric": "wall_time_ms",
            "value": float("inf"),
            "unit": "ratio",
        }
    ]

    errors = validate_result(result, root=tmp_path)

    assert any("comparisons[0].value must be finite" in error for error in errors)
