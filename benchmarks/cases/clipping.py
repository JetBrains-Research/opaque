from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from benchmarks.core import (
    BenchmarkCase,
    CaseRun,
    RunOptions,
    exact_metric,
    sampled_metric,
)
from benchmarks.measurement import device_memory_metrics, measure_callable

if TYPE_CHECKING:
    from collections.abc import Mapping


def _leaves(value: Any) -> list[torch.Tensor]:
    leaves, _ = torch.utils._pytree.tree_flatten(value)
    return [leaf for leaf in leaves if isinstance(leaf, torch.Tensor)]


def _run(config: Mapping[str, Any], options: RunOptions) -> CaseRun:
    from opaque.dpsgd.clipping import clipped_grad

    torch.manual_seed(int(config["seed"]))
    batch_size = int(config["batch_size"])
    input_dim = int(config["input_dim"])
    output_dim = int(config["output_dim"])
    dtype = getattr(torch, str(config["dtype"]))
    params = {
        "weight": torch.randn(
            input_dim, output_dim, device=options.device, dtype=dtype
        ),
        "bias": torch.randn(output_dim, device=options.device, dtype=dtype),
    }
    features = torch.randn(batch_size, input_dim, device=options.device, dtype=dtype)
    targets = torch.randn(batch_size, output_dim, device=options.device, dtype=dtype)

    def loss_fn(parameters, feature, target):
        prediction = feature @ parameters["weight"] + parameters["bias"]
        return (prediction - target).square().mean()

    baseline_leaves: list[torch.Tensor] | None = None
    measurements: list[dict[str, Any]] = []
    for microbatch in config["microbatch_sizes"]:
        clip_fn, state = clipped_grad(
            loss_fn,
            batch_argnums=(1, 2),
            clipping_norm=float(config["clipping_norm"]),
            microbatch_size=microbatch,
            return_aux=False,
        )

        def operation(clip_fn=clip_fn, state=state):
            clipped, _ = clip_fn(params, features, targets, state=state)
            return clipped

        timing = measure_callable(operation, options)
        output = operation()
        output_leaves = _leaves(output.pytree)
        if baseline_leaves is None:
            baseline_leaves = [leaf.detach().clone() for leaf in output_leaves]
        max_abs_error = max(
            (
                (actual - expected).abs().max().item()
                for actual, expected in zip(output_leaves, baseline_leaves, strict=True)
            ),
            default=0.0,
        )
        median_ms = float(sampled_metric(timing.samples_ms, "ms")["value"])
        metrics = {
            "wall_time_ms": sampled_metric(timing.samples_ms, "ms"),
            "examples_per_second": exact_metric(
                batch_size * 1000.0 / median_ms, "example/s"
            ),
            "max_abs_error": exact_metric(max_abs_error, "absolute"),
        }
        metrics.update(device_memory_metrics(timing))
        measurements.append(
            {
                "name": "full_batch"
                if microbatch is None
                else f"microbatch_{microbatch}",
                "parameters": {"microbatch_size": microbatch},
                "metrics": metrics,
            }
        )

    return CaseRun(
        measurements=measurements,
        notes=[
            "CUDA peak bytes are peak allocated bytes; MPS peak bytes are sampled "
            "driver allocation at 0.5 ms intervals; CPU exposes no allocator peak."
        ],
    )


CASE = BenchmarkCase(
    case_id="clipping.microbatch",
    description="Fixed-clipping throughput and peak memory across microbatch sizes.",
    source_files=(
        "benchmarks/cases/clipping.py",
        "benchmarks/core.py",
        "benchmarks/measurement.py",
        "packages/opaque-engine/src/opaque/api/engine/clipping",
    ),
    presets={
        "smoke": {
            "seed": 417,
            "batch_size": 2,
            "input_dim": 4,
            "output_dim": 3,
            "clipping_norm": 1.0,
            "microbatch_sizes": [None, 1],
            "dtype": "float32",
        },
        "reference": {
            "seed": 417,
            "batch_size": 64,
            "input_dim": 512,
            "output_dim": 256,
            "clipping_norm": 1.0,
            "microbatch_sizes": [None, 32, 8, 1],
            "dtype": "float32",
        },
    },
    devices=("cpu", "mps", "cuda"),
    runner=_run,
)

__all__ = ["CASE"]
