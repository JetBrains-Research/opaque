from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from benchmarks.core import (
    BenchmarkCase,
    CaseRun,
    RunOptions,
    exact_metric,
    sampled_metric,
)
from benchmarks.measurement import (
    TimingMeasurement,
    device_memory_metrics,
    measure_callable,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


@contextlib.contextmanager
def _determinism(enabled: bool, seed: int) -> Iterator[None]:
    from opaque.random import key, set_reproducible_pytorch_seed

    previous_algorithms = torch.are_deterministic_algorithms_enabled()
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    previous_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        if enabled:
            set_reproducible_pytorch_seed(key(seed))
        else:
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(False)
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        yield
    finally:
        torch.use_deterministic_algorithms(previous_algorithms)
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark
        if previous_workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_workspace


def _timing_metrics(timing: TimingMeasurement) -> dict[str, dict[str, Any]]:
    metrics = {"wall_time_ms": sampled_metric(timing.samples_ms, "ms")}
    metrics.update(device_memory_metrics(timing))
    return metrics


def _run(config: Mapping[str, Any], options: RunOptions) -> CaseRun:
    seed = int(config["seed"])
    torch.manual_seed(seed)
    dtype = getattr(torch, str(config["dtype"]))
    inputs = torch.randn(
        int(config["batch_size"]),
        int(config["in_channels"]),
        int(config["spatial_size"]),
        int(config["spatial_size"]),
        device=options.device,
        dtype=dtype,
        requires_grad=True,
    )
    weight = torch.randn(
        int(config["out_channels"]),
        int(config["in_channels"]),
        3,
        3,
        device=options.device,
        dtype=dtype,
        requires_grad=True,
    )
    with torch.no_grad():
        reference = F.conv2d(inputs, weight, padding=1).detach()

    measurements: list[dict[str, Any]] = []
    times: dict[str, float] = {}
    for name, enabled in (("default", False), ("deterministic", True)):
        with _determinism(enabled, seed):

            def operation():
                inputs.grad = None
                weight.grad = None
                output = F.conv2d(inputs, weight, padding=1)
                output.square().mean().backward()
                return output

            timing = measure_callable(operation, options)
            with torch.no_grad():
                output = F.conv2d(inputs, weight, padding=1)
                max_abs_error = (output - reference).abs().max().item()
        metrics = _timing_metrics(timing)
        metrics["max_abs_error"] = exact_metric(max_abs_error, "absolute")
        measurements.append(
            {
                "name": name,
                "parameters": {"deterministic_algorithms": enabled},
                "metrics": metrics,
            }
        )
        times[name] = float(metrics["wall_time_ms"]["value"])

    return CaseRun(
        measurements=measurements,
        comparisons=[
            {
                "name": "deterministic_overhead",
                "baseline": "default",
                "candidate": "deterministic",
                "metric": "wall_time_ms",
                "value": (times["deterministic"] / times["default"] - 1.0) * 100.0,
                "unit": "percent",
            }
        ],
        notes=[
            "This measures one convolution workload; deterministic overhead is "
            "operation, shape, device, and software dependent."
        ],
    )


CASE = BenchmarkCase(
    case_id="engine.determinism",
    description="Default versus reproducible PyTorch convolution forward/backward.",
    source_files=(
        "benchmarks/cases/determinism.py",
        "benchmarks/core.py",
        "benchmarks/measurement.py",
        "packages/opaque-engine/src/opaque/api/engine/random",
    ),
    presets={
        "smoke": {
            "seed": 417,
            "batch_size": 2,
            "in_channels": 4,
            "out_channels": 8,
            "spatial_size": 16,
            "dtype": "float32",
        },
        "reference": {
            "seed": 417,
            "batch_size": 16,
            "in_channels": 64,
            "out_channels": 128,
            "spatial_size": 128,
            "dtype": "float32",
        },
    },
    devices=("cpu", "mps", "cuda"),
    runner=_run,
)

__all__ = ["CASE"]
