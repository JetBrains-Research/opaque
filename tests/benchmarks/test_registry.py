from __future__ import annotations

import pytest
import torch
from benchmarks.cases import get_case, list_cases
from benchmarks.core import BenchmarkError, RunOptions


def test_registry_covers_major_performance_surfaces() -> None:
    assert {case.case_id for case in list_cases()} == {
        "accounting.fft",
        "clipping.microbatch",
        "dpftrl.strategy_init",
        "engine.determinism",
        "optimizers.state",
        "patches.fused_linear_cross_entropy",
        "patches.fused_swiglu",
        "patches.moe_dispatch",
        "training.dpsgd",
    }


def test_registry_rejects_unknown_case() -> None:
    with pytest.raises(BenchmarkError, match="Unknown benchmark case"):
        get_case("missing.case")


def test_run_options_validate_counts() -> None:
    with pytest.raises(ValueError, match="repeats"):
        RunOptions(device="cpu", warmup=0, repeats=0)
    with pytest.raises(ValueError, match="warmup"):
        RunOptions(device="cpu", warmup=-1, repeats=1)


def test_optimizer_state_smoke_case_measures_tensor_state() -> None:
    case = get_case("optimizers.state")

    run = case.run(case.presets["smoke"], RunOptions("cpu", warmup=0, repeats=1))

    measurements = {row["name"]: row for row in run.measurements}
    assert measurements["sgd"]["metrics"]["state_bytes"]["value"] == 0
    assert (
        measurements["adadelta_bc"]["metrics"]["state_bytes"]["value"]
        > measurements["adadelta"]["metrics"]["state_bytes"]["value"]
    )
    assert all(
        row["metrics"]["parameter_bytes"]["value"] > 0 for row in measurements.values()
    )


def test_clipping_smoke_case_checks_microbatch_equivalence() -> None:
    case = get_case("clipping.microbatch")

    run = case.run(case.presets["smoke"], RunOptions("cpu", warmup=0, repeats=1))

    assert len(run.measurements) == 2
    assert all(
        row["metrics"]["max_abs_error"]["value"] <= 1e-5 for row in run.measurements
    )


def test_dpftrl_strategy_smoke_case_measures_construction_and_row_norms() -> None:
    case = get_case("dpftrl.strategy_init")

    run = case.run(case.presets["smoke"], RunOptions("cpu", warmup=0, repeats=1))

    assert {row["name"] for row in run.measurements} == {"horizon_8", "horizon_16"}
    assert all(
        row["metrics"]["construction_time_ms"]["value"] > 0
        and row["metrics"]["row_norms_time_ms"]["value"] > 0
        for row in run.measurements
    )


@pytest.mark.slow
def test_accounting_fft_smoke_case_runs_native_rust_comparison() -> None:
    case = get_case("accounting.fft")

    run = case.run(case.presets["smoke"], RunOptions("cpu", warmup=0, repeats=1))

    assert {row["name"] for row in run.measurements} == {
        "real_fft_n64",
        "complex_fft_n64",
        "real_fft_n128",
        "complex_fft_n128",
    }
    assert all(
        row["metrics"]["max_abs_error"]["value"] <= 1e-9 for row in run.measurements
    )


def test_determinism_smoke_case_measures_both_modes() -> None:
    case = get_case("engine.determinism")

    run = case.run(case.presets["smoke"], RunOptions("cpu", warmup=0, repeats=1))

    assert {row["name"] for row in run.measurements} == {
        "default",
        "deterministic",
    }
    assert all(
        row["metrics"]["max_abs_error"]["value"] <= 1e-6 for row in run.measurements
    )


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
def test_moe_dispatch_smoke_case_compares_real_paths() -> None:
    case = get_case("patches.moe_dispatch")

    run = case.run(case.presets["smoke"], RunOptions("mps", warmup=0, repeats=1))

    assert {row["name"] for row in run.measurements} == {
        "dense_e8",
        "grouped_e8",
        "dense_e16",
        "grouped_e16",
    }
    assert all(
        row["metrics"]["max_abs_error"]["value"] <= 1e-4 for row in run.measurements
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_linear_cross_entropy_smoke_case_compares_real_paths() -> None:
    pytest.importorskip("triton")
    case = get_case("patches.fused_linear_cross_entropy")

    run = case.run(case.presets["smoke"], RunOptions("cuda", warmup=0, repeats=1))

    assert {row["name"] for row in run.measurements} == {"pytorch", "opaque"}
    assert all(
        row["metrics"]["loss_abs_error"]["value"] <= 5e-3 for row in run.measurements
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_swiglu_smoke_case_compares_real_paths() -> None:
    pytest.importorskip("triton")
    case = get_case("patches.fused_swiglu")

    run = case.run(case.presets["smoke"], RunOptions("cuda", warmup=0, repeats=1))

    assert {row["name"] for row in run.measurements} == {
        "forward_backward_pytorch",
        "forward_backward_opaque",
        "vmap_grad_pytorch",
        "vmap_grad_opaque",
    }
    assert all(
        row["metrics"]["max_abs_error"]["value"] <= 5e-3 for row in run.measurements
    )
