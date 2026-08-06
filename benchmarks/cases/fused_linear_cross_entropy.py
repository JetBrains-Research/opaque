from __future__ import annotations

from functools import partial
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
    from collections.abc import Callable, Mapping


def _metrics(timing: TimingMeasurement) -> dict[str, dict[str, Any]]:
    metrics = {"wall_time_ms": sampled_metric(timing.samples_ms, "ms")}
    metrics.update(device_memory_metrics(timing))
    return metrics


def _backward(implementation, hidden, weight, labels):
    hidden.grad = None
    weight.grad = None
    loss = implementation(hidden, weight, labels)
    loss.backward()
    return loss


def _run(config: Mapping[str, Any], options: RunOptions) -> CaseRun:
    from opaque.api.patches.kernels.linear_cross_entropy import (
        opaque_linear_cross_entropy_loss,
    )

    batch = int(config["batch_size"])
    sequence = int(config["sequence_length"])
    hidden_dim = int(config["hidden_dim"])
    vocabulary = int(config["vocabulary_size"])
    dtype = getattr(torch, str(config["dtype"]))
    seed = int(config["seed"])

    def pytorch_loss(hidden, weight, labels):
        logits = hidden.float() @ weight.float().T
        return F.cross_entropy(
            logits[..., :-1, :].reshape(-1, vocabulary),
            labels[..., 1:].reshape(-1),
        )

    implementations: tuple[
        tuple[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]],
        ...,
    ] = (
        ("pytorch", pytorch_loss),
        ("opaque", opaque_linear_cross_entropy_loss),
    )
    measurements: list[dict[str, Any]] = []
    losses: dict[str, float] = {}
    for name, implementation in implementations:
        torch.manual_seed(seed)
        hidden = torch.randn(
            batch,
            sequence,
            hidden_dim,
            device=options.device,
            dtype=dtype,
            requires_grad=True,
        )
        weight = torch.randn(
            vocabulary,
            hidden_dim,
            device=options.device,
            dtype=dtype,
            requires_grad=True,
        )
        labels = torch.randint(0, vocabulary, (batch, sequence), device=options.device)

        operation = partial(_backward, implementation, hidden, weight, labels)

        with torch.no_grad():
            loss_value = implementation(hidden, weight, labels).item()
        losses[name] = loss_value
        timing = measure_callable(operation, options)
        metrics = _metrics(timing)
        metrics["loss_abs_error"] = exact_metric(
            abs(loss_value - losses["pytorch"]), "absolute"
        )
        measurements.append(
            {
                "name": name,
                "parameters": {"implementation": name},
                "metrics": metrics,
            }
        )

    pytorch_row, opaque_row = measurements
    comparisons = [
        {
            "name": "opaque_speedup",
            "baseline": "pytorch",
            "candidate": "opaque",
            "metric": "wall_time_ms",
            "value": (
                pytorch_row["metrics"]["wall_time_ms"]["value"]
                / opaque_row["metrics"]["wall_time_ms"]["value"]
            ),
            "unit": "ratio",
        },
        {
            "name": "opaque_peak_memory_reduction",
            "baseline": "pytorch",
            "candidate": "opaque",
            "metric": "peak_device_bytes",
            "value": (
                pytorch_row["metrics"]["peak_device_bytes"]["value"]
                / opaque_row["metrics"]["peak_device_bytes"]["value"]
            ),
            "unit": "ratio",
        },
    ]
    return CaseRun(measurements=measurements, comparisons=comparisons)


CASE = BenchmarkCase(
    case_id="patches.fused_linear_cross_entropy",
    description="PyTorch versus fused linear cross-entropy forward/backward.",
    source_files=(
        "benchmarks/cases/fused_linear_cross_entropy.py",
        "benchmarks/core.py",
        "benchmarks/measurement.py",
        "packages/opaque-patches/src/opaque/api/patches/kernels",
    ),
    presets={
        "smoke": {
            "seed": 417,
            "batch_size": 1,
            "sequence_length": 8,
            "hidden_dim": 16,
            "vocabulary_size": 128,
            "dtype": "bfloat16",
        },
        "reference": {
            "seed": 417,
            "batch_size": 4,
            "sequence_length": 1024,
            "hidden_dim": 3072,
            "vocabulary_size": 128256,
            "dtype": "bfloat16",
        },
    },
    devices=("cuda",),
    runner=_run,
)

__all__ = ["CASE"]
