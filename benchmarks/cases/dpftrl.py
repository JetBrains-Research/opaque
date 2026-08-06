from __future__ import annotations

from typing import TYPE_CHECKING, Any

from benchmarks.core import (
    BenchmarkCase,
    CaseRun,
    RunOptions,
    exact_metric,
    sampled_metric,
)
from benchmarks.measurement import measure_callable

if TYPE_CHECKING:
    from collections.abc import Mapping


def _run(config: Mapping[str, Any], options: RunOptions) -> CaseRun:
    from opaque.api.dpftrl.noise._band_mf import _band_mf_coefficients_cached
    from opaque.dpftrl.noise import band_mf_strategy

    bands = int(config["bands"])
    momentum = float(config["momentum"])
    measurements: list[dict[str, Any]] = []
    for horizon_value in config["horizons"]:
        horizon = int(horizon_value)

        def construct(horizon=horizon):
            _band_mf_coefficients_cached.cache_clear()
            strategy = band_mf_strategy(bands=bands, momentum=momentum)
            return strategy.streaming_matrix(n_steps=horizon)

        construction = measure_callable(construct, options)
        matrix = construct()

        def row_norms_operation(matrix=matrix, horizon=horizon):
            return matrix.row_norms_squared(horizon)

        row_norms = measure_callable(row_norms_operation, options)
        norms = matrix.row_norms_squared(horizon)
        measurements.append(
            {
                "name": f"horizon_{horizon}",
                "parameters": {
                    "horizon": horizon,
                    "bands": bands,
                    "momentum": momentum,
                },
                "metrics": {
                    "construction_time_ms": sampled_metric(
                        construction.samples_ms, "ms"
                    ),
                    "row_norms_time_ms": sampled_metric(row_norms.samples_ms, "ms"),
                    "row_norm_count": exact_metric(norms.numel(), "element"),
                    "max_row_norm_squared": exact_metric(
                        norms.max().item(), "squared_norm"
                    ),
                },
            }
        )

    return CaseRun(
        measurements=measurements,
        notes=[
            "Construction timings clear the BandMF coefficient cache before every "
            "sample; row-norm timings reuse the constructed streaming matrix."
        ],
    )


CASE = BenchmarkCase(
    case_id="dpftrl.strategy_init",
    description="Cold BandMF construction and row-norm scaling by horizon.",
    source_files=(
        "benchmarks/cases/dpftrl.py",
        "benchmarks/core.py",
        "benchmarks/measurement.py",
        "packages/opaque-dpftrl/src/opaque/api/dpftrl/noise",
    ),
    presets={
        "smoke": {"horizons": [8, 16], "bands": 4, "momentum": 0.9},
        "reference": {
            "horizons": [64, 128, 256, 512],
            "bands": 8,
            "momentum": 0.9,
        },
    },
    devices=("cpu",),
    runner=_run,
)

__all__ = ["CASE"]
