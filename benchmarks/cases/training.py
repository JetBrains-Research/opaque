from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks.core import (
    BenchmarkCase,
    BenchmarkError,
    CaseRun,
    RunOptions,
    sampled_metric,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

_ROOT = Path(__file__).resolve().parents[2]
_SUMMARY_PATTERNS = {
    "total_steps": re.compile(r"^\s*Total steps:\s*(\d+)\s*$", re.MULTILINE),
    "samples_per_second": re.compile(
        r"^\s*Throughput:\s*([0-9.]+) samples/s\s*$", re.MULTILINE
    ),
    "steps_per_second": re.compile(r"^\s*Steps/s:\s*([0-9.]+)\s*$", re.MULTILINE),
    "peak_memory_gb": re.compile(r"^\s*Peak memory:\s*([0-9.]+) GB\s*$", re.MULTILINE),
}


def parse_training_summary(output: str) -> dict[str, float | int]:
    matches = {
        name: pattern.search(output) for name, pattern in _SUMMARY_PATTERNS.items()
    }
    missing = [name for name, match in matches.items() if match is None]
    if missing:
        raise BenchmarkError(f"Training output is missing final metrics: {missing}")
    return {
        "total_steps": int(matches["total_steps"].group(1)),  # type: ignore[union-attr]
        "samples_per_second": float(
            matches["samples_per_second"].group(1)  # type: ignore[union-attr]
        ),
        "steps_per_second": float(
            matches["steps_per_second"].group(1)  # type: ignore[union-attr]
        ),
        "peak_memory_gb": float(
            matches["peak_memory_gb"].group(1)  # type: ignore[union-attr]
        ),
    }


def _execute(config: Mapping[str, Any]) -> dict[str, float | int]:
    command = [
        "uv",
        "run",
        "python",
        "examples/train_dpsgd.py",
        "--preset",
        str(config["preset"]),
        "--stop-at-step",
        str(config["stop_at_step"]),
        "--no-wandb",
        *[str(argument) for argument in config.get("extra_args", [])],
    ]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    try:
        completed = subprocess.run(
            command,
            cwd=_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        stderr = getattr(error, "stderr", "")
        raise BenchmarkError(f"DP-SGD training benchmark failed: {stderr}") from error
    return parse_training_summary(completed.stdout)


def _run(config: Mapping[str, Any], options: RunOptions) -> CaseRun:
    for _ in range(options.warmup):
        _execute(config)
    runs = [_execute(config) for _ in range(options.repeats)]
    total_steps = {int(run["total_steps"]) for run in runs}
    if len(total_steps) != 1:
        raise BenchmarkError(
            f"Training repeats produced inconsistent steps: {total_steps}"
        )
    return CaseRun(
        measurements=[
            {
                "name": str(config["preset"]),
                "parameters": dict(config),
                "metrics": {
                    "samples_per_second": sampled_metric(
                        [float(run["samples_per_second"]) for run in runs],
                        "sample/s",
                    ),
                    "steps_per_second": sampled_metric(
                        [float(run["steps_per_second"]) for run in runs], "step/s"
                    ),
                    "peak_memory_gb": sampled_metric(
                        [float(run["peak_memory_gb"]) for run in runs], "GB"
                    ),
                },
            }
        ],
        notes=[
            "Metrics come from the training script's synchronized PerfState summary."
        ],
    )


CASE = BenchmarkCase(
    case_id="training.dpsgd",
    description="End-to-end DP-SGD throughput and peak accelerator memory.",
    source_files=(
        "benchmarks/cases/training.py",
        "benchmarks/core.py",
        "examples/train_dpsgd.py",
        "examples/common",
        "packages/opaque-dpsgd/src",
        "packages/opaque-engine/src",
        "packages/opaque-optimizers/src",
        "packages/opaque-patches/src",
        "packages/opaque-transformers/src",
    ),
    presets={
        "smoke": {"preset": "smoke", "stop_at_step": 2, "extra_args": []},
        "reference": {
            "preset": "mellum-kstack",
            "stop_at_step": 20,
            "extra_args": [],
        },
    },
    devices=("cuda",),
    runner=_run,
)

__all__ = ["CASE", "parse_training_summary"]
