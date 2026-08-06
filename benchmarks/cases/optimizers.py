from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import torch

from benchmarks.core import (
    BenchmarkCase,
    CaseRun,
    RunOptions,
    exact_metric,
    sampled_metric,
)
from benchmarks.measurement import measure_callable, tensor_storage_bytes

if TYPE_CHECKING:
    from collections.abc import Mapping


def _run(config: Mapping[str, Any], options: RunOptions) -> CaseRun:
    from opaque.optimizers import adadelta, adafactor, adamw, lion, sgd

    rows = int(config["matrix_rows"])
    columns = int(config["matrix_columns"])
    vector = int(config["vector_size"])
    dtype = getattr(torch, str(config["dtype"]))
    params = {
        "matrix": torch.zeros(rows, columns, device=options.device, dtype=dtype),
        "vector": torch.zeros(vector, device=options.device, dtype=dtype),
    }
    parameter_bytes = tensor_storage_bytes(params)
    factories = {
        "sgd": lambda: sgd(lr=0.1),
        "sgd_momentum": lambda: sgd(lr=0.1, momentum=0.9),
        "lion": lion,
        "adafactor": adafactor,
        "adamw": adamw,
        "adamw_bc": lambda: adamw(noise_bias_correction=True),
        "adadelta": adadelta,
        "adadelta_bc": lambda: adadelta(noise_bias_correction=True),
    }

    measurements: list[dict[str, Any]] = []
    for name, factory in factories.items():
        optimizer = factory()
        timing = measure_callable(partial(optimizer.init, params), options)
        state = optimizer.init(params)
        state_bytes = tensor_storage_bytes(state)
        measurements.append(
            {
                "name": name,
                "parameters": {
                    "matrix_shape": [rows, columns],
                    "vector_size": vector,
                    "dtype": str(config["dtype"]),
                },
                "metrics": {
                    "parameter_bytes": exact_metric(parameter_bytes, "byte"),
                    "state_bytes": exact_metric(state_bytes, "byte"),
                    "state_bytes_per_parameter_byte": exact_metric(
                        state_bytes / parameter_bytes, "ratio"
                    ),
                    "initialization_time_ms": sampled_metric(timing.samples_ms, "ms"),
                },
            }
        )

    return CaseRun(
        measurements=measurements,
        notes=[
            "State size counts unique tensor storage only; Python object overhead "
            "is intentionally excluded."
        ],
    )


CASE = BenchmarkCase(
    case_id="optimizers.state",
    description="Initialized tensor-state size and initialization time by optimizer.",
    source_files=(
        "benchmarks/cases/optimizers.py",
        "benchmarks/core.py",
        "benchmarks/measurement.py",
        "packages/opaque-optimizers/src/opaque/api/optimizers",
    ),
    presets={
        "smoke": {
            "matrix_rows": 4,
            "matrix_columns": 3,
            "vector_size": 3,
            "dtype": "float32",
        },
        "reference": {
            "matrix_rows": 2048,
            "matrix_columns": 2048,
            "vector_size": 2048,
            "dtype": "float32",
        },
    },
    devices=("cpu", "mps", "cuda"),
    runner=_run,
)

__all__ = ["CASE"]
