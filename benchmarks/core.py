from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_SOURCE_FILES = ("pyproject.toml", "uv.lock")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class BenchmarkError(RuntimeError):
    """Raised when a benchmark cannot be selected or executed."""


@dataclass(frozen=True)
class RunOptions:
    device: str
    warmup: int = 3
    repeats: int = 10

    def __post_init__(self) -> None:
        if self.warmup < 0:
            raise ValueError("warmup must be >= 0")
        if self.repeats <= 0:
            raise ValueError("repeats must be > 0")


@dataclass(frozen=True)
class CaseRun:
    measurements: list[dict[str, Any]]
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    description: str
    source_files: tuple[str, ...]
    presets: Mapping[str, Mapping[str, Any]]
    devices: tuple[str, ...]
    runner: Callable[[Mapping[str, Any], RunOptions], CaseRun]

    def run(self, config: Mapping[str, Any], options: RunOptions) -> CaseRun:
        if options.device not in self.devices:
            choices = ", ".join(self.devices)
            raise BenchmarkError(
                f"Case {self.case_id!r} does not support device "
                f"{options.device!r}; choose one of: {choices}"
            )
        return self.runner(config, options)


def summarize_samples(samples: Sequence[float]) -> dict[str, float | int]:
    values = [float(value) for value in samples]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("samples must contain at least one finite value")
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def sampled_metric(samples: Sequence[float], unit: str) -> dict[str, Any]:
    values = [float(value) for value in samples]
    summary = summarize_samples(values)
    return {
        "value": summary["median"],
        "unit": unit,
        "statistic": "median",
        "samples": values,
        "summary": summary,
    }


def exact_metric(value: float | int, unit: str) -> dict[str, Any]:
    if not math.isfinite(float(value)):
        raise ValueError("metric value must be finite")
    return {"value": value, "unit": unit, "statistic": "exact"}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files(root: Path, relative: str) -> list[Path]:
    path = root / relative
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            child
            for child in path.rglob("*")
            if child.is_file()
            and "__pycache__" not in child.parts
            and child.suffix not in {".pyc", ".pyo"}
        )
    raise BenchmarkError(f"Benchmark source does not exist: {relative}")


def capture_sources(root: Path, paths: Sequence[str]) -> dict[str, str]:
    sources: dict[str, str] = {}
    for relative in sorted(set(paths)):
        for path in _source_files(root, relative):
            source_path = path.relative_to(root).as_posix()
            sources[source_path] = _file_sha256(path)
    if not sources:
        raise BenchmarkError("A benchmark case must declare at least one source file")
    return sources


def benchmark_source_paths(case: BenchmarkCase) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*case.source_files, *DEFAULT_SOURCE_FILES)))


def source_digest(sources: Mapping[str, str]) -> str:
    payload = json.dumps(dict(sorted(sources.items())), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.rstrip("\n")


def _parse_dirty_paths(status: str) -> list[str]:
    dirty_paths = []
    for line in status.splitlines():
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        dirty_paths.append(path)
    return dirty_paths


def _git_provenance(root: Path) -> dict[str, Any]:
    try:
        commit = _git_output(root, "rev-parse", "HEAD")
        status = _git_output(root, "status", "--porcelain=v1", "--untracked-files=all")
    except (OSError, subprocess.CalledProcessError) as error:
        raise BenchmarkError("Benchmarks must run inside a Git worktree") from error
    dirty_paths = _parse_dirty_paths(status)
    return {
        "commit": commit,
        "dirty": bool(dirty_paths),
        "dirty_paths": dirty_paths,
    }


def _cpu_model() -> str:
    model = platform.processor().strip()
    if sys.platform == "darwin":
        with contextlib.suppress(OSError, subprocess.CalledProcessError):
            model = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    return model or platform.machine() or "unknown"


def _physical_memory_bytes() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return 0


def _accelerator(device: str) -> dict[str, Any] | None:
    if device == "cpu":
        return None
    try:
        import torch
    except ImportError:
        return {"type": device, "available": False}
    if device == "cuda" and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        return {
            "type": "cuda",
            "name": properties.name,
            "memory_bytes": properties.total_memory,
            "capability": list(torch.cuda.get_device_capability(0)),
            "runtime": torch.version.cuda,
        }
    if device == "mps" and torch.backends.mps.is_available():
        return {
            "type": "mps",
            "name": _cpu_model(),
            "memory_bytes": int(torch.mps.recommended_max_memory()),
            "runtime": platform.mac_ver()[0],
        }
    return {"type": device, "available": False}


def collect_provenance(
    root: Path, *, command: Sequence[str], device: str
) -> dict[str, Any]:
    packages: dict[str, str] = {}
    for distribution in (
        "opaque",
        "opaque-accounting",
        "opaque-engine",
        "opaque-optimizers",
        "opaque-patches",
        "torch",
        "torchopt",
        "triton",
    ):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    tools: dict[str, str] = {}
    for tool in ("cargo", "rustc"):
        try:
            tools[tool] = subprocess.run(
                [tool, "--version"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            continue
    return {
        "timestamp_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "command": list(command),
        "git": _git_provenance(root),
        "hardware": {
            "machine": platform.machine() or "unknown",
            "cpu": _cpu_model(),
            "memory_bytes": _physical_memory_bytes(),
            "accelerator": _accelerator(device),
        },
        "software": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": packages,
            "tools": tools,
        },
    }


def build_result(
    case: BenchmarkCase,
    config: Mapping[str, Any],
    run: CaseRun,
    *,
    root: Path,
    command: Sequence[str],
    device: str,
) -> dict[str, Any]:
    sources = capture_sources(root, benchmark_source_paths(case))
    return {
        "schema_version": SCHEMA_VERSION,
        "case": {
            "id": case.case_id,
            "description": case.description,
            "config": dict(config),
            "sources": sources,
            "source_digest": source_digest(sources),
        },
        "provenance": collect_provenance(root, command=command, device=device),
        "measurements": run.measurements,
        "comparisons": run.comparisons,
        "notes": run.notes,
    }


def _nested(result: Mapping[str, Any], path: str, errors: list[str]) -> Any:
    value: Any = result
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            errors.append(f"missing {path}")
            return None
        value = value[part]
    return value


def validate_result(
    result: Mapping[str, Any], *, root: Path, check_sources: bool = True
) -> list[str]:
    errors: list[str] = []
    if result.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")

    case_id = _nested(result, "case.id", errors)
    if not isinstance(case_id, str) or not case_id:
        errors.append("case.id must be a non-empty string")
    config = _nested(result, "case.config", errors)
    if not isinstance(config, Mapping):
        errors.append("case.config must be an object")
    sources = _nested(result, "case.sources", errors)
    digest = _nested(result, "case.source_digest", errors)
    if not isinstance(sources, Mapping) or not sources:
        errors.append("case.sources must be a non-empty object")
    else:
        malformed = [
            path
            for path, value in sources.items()
            if not isinstance(path, str)
            or not isinstance(value, str)
            or not _SHA256_RE.fullmatch(value)
        ]
        if malformed:
            errors.append(f"case.sources contains malformed entries: {malformed}")
        elif digest != source_digest(sources):
            errors.append("case.source_digest does not match case.sources")
        if check_sources and not malformed:
            for relative, expected in sources.items():
                path = root / relative
                if not path.is_file():
                    errors.append(f"missing source {relative}")
                elif _file_sha256(path) != expected:
                    errors.append(f"stale source {relative}")

    command = _nested(result, "provenance.command", errors)
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(part, str) and part for part in command)
    ):
        errors.append("provenance.command must be a non-empty string array")
    commit = _nested(result, "provenance.git.commit", errors)
    if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
        errors.append("provenance.git.commit must be a full Git commit")
    for path in (
        "provenance.timestamp_utc",
        "provenance.hardware.machine",
        "provenance.hardware.cpu",
        "provenance.software.platform",
        "provenance.software.python",
    ):
        value = _nested(result, path, errors)
        if not isinstance(value, str) or not value:
            errors.append(f"{path} must be a non-empty string")
    memory = _nested(result, "provenance.hardware.memory_bytes", errors)
    if not isinstance(memory, int) or memory <= 0:
        errors.append("provenance.hardware.memory_bytes must be positive")
    packages = _nested(result, "provenance.software.packages", errors)
    if not isinstance(packages, Mapping) or not packages:
        errors.append("provenance.software.packages must be a non-empty object")

    measurements = result.get("measurements")
    if not isinstance(measurements, list) or not measurements:
        errors.append("measurements must be a non-empty array")
        return errors
    for row_index, row in enumerate(measurements):
        if not isinstance(row, Mapping):
            errors.append(f"measurements[{row_index}] must be an object")
            continue
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping) or not metrics:
            errors.append(f"measurements[{row_index}].metrics must be non-empty")
            continue
        for metric_name, metric in metrics.items():
            prefix = f"measurements[{row_index}].metrics.{metric_name}"
            if not isinstance(metric, Mapping):
                errors.append(f"{prefix} must be an object")
                continue
            value = metric.get("value")
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                errors.append(f"{prefix}.value must be finite")
            if not isinstance(metric.get("unit"), str) or not metric.get("unit"):
                errors.append(f"{prefix}.unit must be a non-empty string")
            samples = metric.get("samples")
            if samples is not None and (
                not isinstance(samples, list)
                or not samples
                or not all(
                    isinstance(sample, (int, float)) and math.isfinite(float(sample))
                    for sample in samples
                )
            ):
                errors.append(f"{prefix}.samples must contain finite numbers")
    comparisons = result.get("comparisons", [])
    if not isinstance(comparisons, list):
        errors.append("comparisons must be an array")
    else:
        for index, comparison in enumerate(comparisons):
            prefix = f"comparisons[{index}]"
            if not isinstance(comparison, Mapping):
                errors.append(f"{prefix} must be an object")
                continue
            for field in ("name", "baseline", "candidate", "metric", "unit"):
                value = comparison.get(field)
                if not isinstance(value, str) or not value:
                    errors.append(f"{prefix}.{field} must be a non-empty string")
            value = comparison.get("value")
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                errors.append(f"{prefix}.value must be finite")
    return errors


def write_result(result: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_result(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"Cannot read benchmark result {path}: {error}") from error
    if not isinstance(loaded, dict):
        raise BenchmarkError(f"Benchmark result {path} must contain a JSON object")
    return loaded


__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_SOURCE_FILES",
    "BenchmarkCase",
    "BenchmarkError",
    "CaseRun",
    "RunOptions",
    "build_result",
    "benchmark_source_paths",
    "capture_sources",
    "collect_provenance",
    "exact_metric",
    "load_result",
    "sampled_metric",
    "source_digest",
    "summarize_samples",
    "validate_result",
    "write_result",
]
