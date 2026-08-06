from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from benchmarks.core import (
    BenchmarkCase,
    BenchmarkError,
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


def _timing_metrics(timing: TimingMeasurement) -> dict[str, dict[str, Any]]:
    metrics = {"wall_time_ms": sampled_metric(timing.samples_ms, "ms")}
    metrics.update(device_memory_metrics(timing))
    return metrics


def _inputs(config: Mapping[str, Any], experts: int, device: str):
    torch.manual_seed(int(config["seed"]) + experts)
    tokens = int(config["tokens"])
    hidden = int(config["hidden_dim"])
    intermediate = int(config["intermediate_dim"])
    top_k = int(config["top_k"])
    dtype = getattr(torch, str(config["dtype"]))
    x = torch.randn(tokens, hidden, device=device, dtype=dtype)
    gate_up = torch.randn(
        experts, 2 * intermediate, hidden, device=device, dtype=dtype
    ) * float(config["weight_scale"])
    down = torch.randn(
        experts, hidden, intermediate, device=device, dtype=dtype
    ) * float(config["weight_scale"])
    logits = torch.randn(tokens, experts, device=device, dtype=torch.float32)
    top_k_weights, top_k_index = torch.topk(F.softmax(logits, dim=-1), top_k, dim=-1)
    top_k_weights = (top_k_weights / top_k_weights.sum(-1, keepdim=True)).to(dtype)
    return x, gate_up, down, top_k_index, top_k_weights


def _forward_backward(implementation, x, gate_up, down, top_k_index, top_k_weights):
    x.grad = None
    gate_up.grad = None
    down.grad = None
    output = implementation(x, gate_up, down, top_k_index, top_k_weights)
    output.square().mean().backward()
    return output


def _run(config: Mapping[str, Any], options: RunOptions) -> CaseRun:
    from opaque.api.patches.kernels._grouped_moe import (
        Opaque_GroupedMoE,
        grouped_mm_available,
    )
    from opaque.api.patches.kernels.moe import Opaque_MoE

    if not grouped_mm_available():
        raise BenchmarkError("torch._grouped_mm is unavailable on this PyTorch build")
    implementations: tuple[tuple[str, Callable[..., torch.Tensor]], ...] = (
        ("dense", Opaque_MoE.apply),
        ("grouped", Opaque_GroupedMoE.apply),
    )
    measurements: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for expert_value in config["expert_counts"]:
        experts = int(expert_value)
        base_inputs = _inputs(config, experts, options.device)
        with torch.no_grad():
            reference = Opaque_MoE.apply(*base_inputs).detach()
        rows_for_expert: dict[str, dict[str, Any]] = {}
        for name, implementation in implementations:
            x, gate_up, down, top_k_index, top_k_weights = (
                tensor.detach().clone() for tensor in base_inputs
            )
            x.requires_grad_(True)
            gate_up.requires_grad_(True)
            down.requires_grad_(True)

            operation = partial(
                _forward_backward,
                implementation,
                x,
                gate_up,
                down,
                top_k_index,
                top_k_weights,
            )

            timing = measure_callable(operation, options)
            with torch.no_grad():
                output = implementation(x, gate_up, down, top_k_index, top_k_weights)
                max_abs_error = (output - reference).abs().max().item()
            metrics = _timing_metrics(timing)
            metrics["max_abs_error"] = exact_metric(max_abs_error, "absolute")
            row = {
                "name": f"{name}_e{experts}",
                "parameters": {"implementation": name, "experts": experts},
                "metrics": metrics,
            }
            measurements.append(row)
            rows_for_expert[name] = row
        dense_ms = rows_for_expert["dense"]["metrics"]["wall_time_ms"]["value"]
        grouped_ms = rows_for_expert["grouped"]["metrics"]["wall_time_ms"]["value"]
        comparisons.append(
            {
                "name": f"grouped_speedup_e{experts}",
                "baseline": f"dense_e{experts}",
                "candidate": f"grouped_e{experts}",
                "metric": "wall_time_ms",
                "value": dense_ms / grouped_ms,
                "unit": "ratio",
            }
        )

    return CaseRun(
        measurements=measurements,
        comparisons=comparisons,
        notes=[
            "Both paths include forward and backward. The grouped path is invoked "
            "directly so counts below the production dispatch threshold are measured."
        ],
    )


CASE = BenchmarkCase(
    case_id="patches.moe_dispatch",
    description="Dense versus grouped-MoE forward/backward crossover on MPS.",
    source_files=(
        "benchmarks/cases/moe.py",
        "benchmarks/core.py",
        "benchmarks/measurement.py",
        "packages/opaque-patches/src/opaque/api/patches/kernels",
    ),
    presets={
        "smoke": {
            "seed": 417,
            "expert_counts": [8, 16],
            "tokens": 16,
            "hidden_dim": 32,
            "intermediate_dim": 16,
            "top_k": 2,
            "weight_scale": 0.05,
            "dtype": "float32",
        },
        "reference": {
            "seed": 417,
            "expert_counts": [8, 16, 32, 64, 128],
            "tokens": 256,
            "hidden_dim": 256,
            "intermediate_dim": 128,
            "top_k": 2,
            "weight_scale": 0.05,
            "dtype": "float32",
        },
    },
    devices=("mps",),
    runner=_run,
)

__all__ = ["CASE"]
