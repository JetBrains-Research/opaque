from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch.func import grad, vmap

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


def _max_error(actual: Any, expected: Any) -> float:
    actual_leaves, _ = torch.utils._pytree.tree_flatten(actual)
    expected_leaves, _ = torch.utils._pytree.tree_flatten(expected)
    return max(
        (
            (left.float() - right.float()).abs().max().item()
            for left, right in zip(actual_leaves, expected_leaves, strict=True)
        ),
        default=0.0,
    )


def _forward_backward(implementation, gate, up):
    gate.grad = None
    up.grad = None
    output = implementation(gate, up)
    output.mean().backward()
    return output


def _loss(implementation, gate, up):
    return implementation(gate, up).mean()


def _run(config: Mapping[str, Any], options: RunOptions) -> CaseRun:
    from opaque.api.patches.kernels.swiglu import opaque_swiglu

    implementations: tuple[
        tuple[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]], ...
    ] = (
        ("pytorch", lambda gate, up: F.silu(gate) * up),
        ("opaque", opaque_swiglu),
    )
    batch = int(config["batch_size"])
    sequence = int(config["sequence_length"])
    intermediate = int(config["intermediate_dim"])
    vmap_batch = int(config["vmap_batch"])
    dtype = getattr(torch, str(config["dtype"]))
    seed = int(config["seed"])
    measurements: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []

    for mode in ("forward_backward", "vmap_grad"):
        shape = (batch, sequence, intermediate)
        if mode == "vmap_grad":
            shape = (vmap_batch, *shape)
        torch.manual_seed(seed)
        gate_base = torch.randn(*shape, device=options.device, dtype=dtype)
        up_base = torch.randn(*shape, device=options.device, dtype=dtype)
        reference: Any = None
        rows: dict[str, dict[str, Any]] = {}
        for name, implementation in implementations:
            gate = gate_base.detach().clone().requires_grad_(mode == "forward_backward")
            up = up_base.detach().clone().requires_grad_(mode == "forward_backward")
            if mode == "forward_backward":
                operation = partial(_forward_backward, implementation, gate, up)
            else:
                transformed = vmap(grad(partial(_loss, implementation), argnums=(0, 1)))
                operation = partial(transformed, gate, up)

            timing = measure_callable(operation, options)
            output = operation()
            if reference is None:
                reference = torch.utils._pytree.tree_map(
                    lambda tensor: tensor.detach().clone(), output
                )
            metrics = _metrics(timing)
            metrics["max_abs_error"] = exact_metric(
                _max_error(output, reference), "absolute"
            )
            row = {
                "name": f"{mode}_{name}",
                "parameters": {"mode": mode, "implementation": name},
                "metrics": metrics,
            }
            rows[name] = row
            measurements.append(row)
        for metric_name, comparison_name in (
            ("wall_time_ms", "speedup"),
            ("peak_device_bytes", "peak_memory_reduction"),
        ):
            comparisons.append(
                {
                    "name": f"{mode}_{comparison_name}",
                    "baseline": f"{mode}_pytorch",
                    "candidate": f"{mode}_opaque",
                    "metric": metric_name,
                    "value": (
                        rows["pytorch"]["metrics"][metric_name]["value"]
                        / rows["opaque"]["metrics"][metric_name]["value"]
                    ),
                    "unit": "ratio",
                }
            )
    return CaseRun(measurements=measurements, comparisons=comparisons)


CASE = BenchmarkCase(
    case_id="patches.fused_swiglu",
    description="PyTorch versus fused SwiGLU runtime and peak memory.",
    source_files=(
        "benchmarks/cases/fused_swiglu.py",
        "benchmarks/core.py",
        "benchmarks/measurement.py",
        "packages/opaque-patches/src/opaque/api/patches/kernels",
    ),
    presets={
        "smoke": {
            "seed": 417,
            "batch_size": 1,
            "sequence_length": 4,
            "intermediate_dim": 128,
            "vmap_batch": 2,
            "dtype": "bfloat16",
        },
        "reference": {
            "seed": 417,
            "batch_size": 4,
            "sequence_length": 1024,
            "intermediate_dim": 8256,
            "vmap_batch": 4,
            "dtype": "bfloat16",
        },
    },
    devices=("cuda",),
    runner=_run,
)

__all__ = ["CASE"]
