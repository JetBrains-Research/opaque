from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks.core import (
    BenchmarkCase,
    BenchmarkError,
    CaseRun,
    RunOptions,
    exact_metric,
    sampled_metric,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_PREFIX = "OPAQUE_BENCHMARK_JSON="


def _run(config: Mapping[str, Any], options: RunOptions) -> CaseRun:
    sizes = [int(size) for size in config["sizes"]]
    command = [
        "cargo",
        "bench",
        "--quiet",
        "--manifest-path",
        "packages/opaque-accounting/Cargo.toml",
        "--bench",
        "fft",
        "--",
        "--sizes",
        ",".join(str(size) for size in sizes),
        "--warmup",
        str(options.warmup),
        "--repeats",
        str(options.repeats),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        stderr = getattr(error, "stderr", "")
        raise BenchmarkError(f"Native Rust FFT benchmark failed: {stderr}") from error
    payload_line = next(
        (
            line
            for line in completed.stdout.splitlines()
            if line.startswith(_OUTPUT_PREFIX)
        ),
        None,
    )
    if payload_line is None:
        raise BenchmarkError("Native Rust FFT benchmark produced no JSON payload")
    try:
        payload = json.loads(payload_line.removeprefix(_OUTPUT_PREFIX))
    except json.JSONDecodeError as error:
        raise BenchmarkError(
            "Native Rust FFT benchmark produced invalid JSON"
        ) from error

    measurements: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for row in payload.get("measurements", []):
        length = int(row["length"])
        error = float(row["max_abs_error"])
        real_metric = sampled_metric(row["real_samples_ms"], "ms")
        complex_metric = sampled_metric(row["complex_samples_ms"], "ms")
        for name, metric in (
            ("real_fft", real_metric),
            ("complex_fft", complex_metric),
        ):
            measurements.append(
                {
                    "name": f"{name}_n{length}",
                    "parameters": {"implementation": name, "length": length},
                    "metrics": {
                        "wall_time_ms": metric,
                        "max_abs_error": exact_metric(error, "absolute"),
                    },
                }
            )
        comparisons.append(
            {
                "name": f"real_fft_speedup_n{length}",
                "baseline": f"complex_fft_n{length}",
                "candidate": f"real_fft_n{length}",
                "metric": "wall_time_ms",
                "value": complex_metric["value"] / real_metric["value"],
                "unit": "ratio",
            }
        )
    if len(measurements) != 2 * len(sizes):
        raise BenchmarkError("Native Rust FFT benchmark returned incomplete sizes")
    return CaseRun(
        measurements=measurements,
        comparisons=comparisons,
        notes=["The real and complex implementations use cached FFT planners."],
    )


CASE = BenchmarkCase(
    case_id="accounting.fft",
    description="Native Rust real-FFT versus complex-FFT convolution throughput.",
    source_files=(
        "Cargo.lock",
        "benchmarks/cases/accounting.py",
        "benchmarks/core.py",
        "packages/opaque-accounting/Cargo.toml",
        "packages/opaque-accounting/benches/fft.rs",
        "packages/opaque-accounting/src/numerics",
    ),
    presets={
        "smoke": {"sizes": [64, 128]},
        "reference": {"sizes": [1024, 4096, 16384, 65536]},
    },
    devices=("cpu",),
    runner=_run,
)

__all__ = ["CASE"]
