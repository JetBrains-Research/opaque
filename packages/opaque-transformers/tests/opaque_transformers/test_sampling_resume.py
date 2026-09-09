"""Checkpoint sampling-law validation."""

from __future__ import annotations

import pytest

import opaque.accounting as acc
import opaque.dpsgd.accounting as dpsgd_acc
from opaque.api.transformers.trainer._sampling_resume import (
    KOutOfTSamplingLaw,
    PoissonSamplingLaw,
    validate_dataset_schedule,
    validate_distributed_resume,
    validate_k_out_of_t_checkpoint,
    validate_poisson_checkpoint,
)
from opaque.exceptions import CheckpointError


def _sampler_state(*, q: float = 0.1, cap: int | None = None, n: int = 100):
    return {
        "sample_rate": q,
        "truncated_batch_size": cap,
        "num_samples": n,
    }


def _step(*, q: float = 0.1, cap: int | None = None, n: int = 100, sigma=1.0):
    kwargs = {"truncated_batch_size": cap, "dataset_size": n} if cap is not None else {}
    return dpsgd_acc.poisson(dpsgd_acc.gaussian(sigma), q, **kwargs)


def _validate(process, *, q=0.1, cap=None, n=100, steps=4):
    validate_poisson_checkpoint(
        sampler_state=_sampler_state(q=q, cap=cap, n=n),
        runtime_sample_rate=q,
        current_law=PoissonSamplingLaw(q, cap, n),
        accounted_process=process,
        global_step=steps,
    )


def _k_out_of_t_state(*, k=2, t=8, allocation="block", n=100):
    return {"k": k, "t": t, "allocation": allocation, "num_samples": n}


def _k_out_of_t_horizon(*, k=2, t=8, allocation="block", sigma=1.0):
    return dpsgd_acc.k_out_of_t(
        dpsgd_acc.gaussian(sigma),
        k=k,
        t=t,
        allocation=allocation,
    )


def test_accepts_one_sampling_law_across_noise_phases():
    process = acc.cached(_step(sigma=0.8)) * 2 | acc.cached(_step(sigma=1.2)) * 2

    _validate(process)


@pytest.mark.parametrize(
    ("current", "match"),
    [
        (PoissonSamplingLaw(0.05, None, 100), "resolved sampling law"),
        (PoissonSamplingLaw(0.1, 10, 100), "resolved sampling law"),
        (PoissonSamplingLaw(0.1, None, 200), "resolved sampling law"),
    ],
)
def test_rejects_current_sampling_law_drift(current, match):
    with pytest.raises(CheckpointError, match=match):
        validate_poisson_checkpoint(
            sampler_state=_sampler_state(),
            runtime_sample_rate=0.1,
            current_law=current,
            accounted_process=_step() * 4,
            global_step=4,
        )


def test_compares_sample_rates_exactly():
    with pytest.raises(CheckpointError, match="resolved sampling law"):
        validate_poisson_checkpoint(
            sampler_state=_sampler_state(),
            runtime_sample_rate=0.1,
            current_law=PoissonSamplingLaw(0.10000001, None, 100),
            accounted_process=_step() * 4,
            global_step=4,
        )


def test_rejects_buried_accountant_drift():
    cap_10 = acc.cached(_step(cap=10))
    cap_20 = acc.cached(_step(cap=20))
    process = cap_10 * 8 | cap_20 * 16 | cap_10 * 4

    with pytest.raises(CheckpointError, match="accountant contains a different"):
        _validate(process, cap=10, steps=28)


@pytest.mark.parametrize("accounted_steps", [3, 5])
def test_rejects_accounting_release_count_drift(accounted_steps):
    with pytest.raises(CheckpointError, match="release count"):
        _validate(_step() * accounted_steps, steps=4)


def test_rejects_runtime_sampler_disagreement():
    with pytest.raises(CheckpointError, match="runtime sample rate"):
        validate_poisson_checkpoint(
            sampler_state=_sampler_state(),
            runtime_sample_rate=0.2,
            current_law=PoissonSamplingLaw(0.1, None, 100),
            accounted_process=_step() * 4,
            global_step=4,
        )


@pytest.mark.parametrize("global_step", [True, 4.5, "4"])
def test_rejects_coercible_accounted_step_count(global_step):
    with pytest.raises(CheckpointError, match="global_step"):
        validate_poisson_checkpoint(
            sampler_state=_sampler_state(),
            runtime_sample_rate=0.1,
            current_law=PoissonSamplingLaw(0.1, None, 100),
            accounted_process=_step() * 4,
            global_step=global_step,
        )


def test_rejects_shared_distributed_sampler_state_before_any_release():
    with pytest.raises(CheckpointError, match="rank-specific sampler state"):
        validate_distributed_resume(
            saved_world_size=2,
            current_world_size=2,
        )


def test_accepts_single_process_resume():
    validate_distributed_resume(saved_world_size=1, current_world_size=1)


def test_rejects_distributed_topology_drift():
    with pytest.raises(CheckpointError, match="topology changed"):
        validate_distributed_resume(
            saved_world_size=2,
            current_world_size=1,
        )


def test_rejects_invalid_distributed_topology():
    with pytest.raises(CheckpointError, match=r"world_size.*>= 1"):
        validate_distributed_resume(saved_world_size=0, current_world_size=1)


def test_requires_matching_dataset_schedule_identity():
    validate_dataset_schedule(saved_identity="train-v1", current_identity="train-v1")

    with pytest.raises(CheckpointError, match=r"requires.*dataset_schedule_id"):
        validate_dataset_schedule(saved_identity=None, current_identity=None)
    with pytest.raises(CheckpointError, match="schedule changed"):
        validate_dataset_schedule(
            saved_identity="train-v1",
            current_identity="train-v2",
        )


def test_accepts_one_k_out_of_t_horizon():
    validate_k_out_of_t_checkpoint(
        sampler_state=_k_out_of_t_state(),
        current_law=KOutOfTSamplingLaw(2, 8, "block", 100),
        accounted_process=acc.cached(_k_out_of_t_horizon()),
    )


@pytest.mark.parametrize(
    "current",
    [
        KOutOfTSamplingLaw(1, 8, "block", 100),
        KOutOfTSamplingLaw(2, 9, "block", 100),
        KOutOfTSamplingLaw(2, 8, "total", 100),
        KOutOfTSamplingLaw(2, 8, "block", 200),
    ],
)
def test_rejects_k_out_of_t_law_drift(current):
    with pytest.raises(CheckpointError, match="resolved sampling law"):
        validate_k_out_of_t_checkpoint(
            sampler_state=_k_out_of_t_state(),
            current_law=current,
            accounted_process=_k_out_of_t_horizon(),
        )


def test_rejects_k_out_of_t_accountant_drift():
    with pytest.raises(CheckpointError, match="different K-out-of-T law"):
        validate_k_out_of_t_checkpoint(
            sampler_state=_k_out_of_t_state(),
            current_law=KOutOfTSamplingLaw(2, 8, "block", 100),
            accounted_process=_k_out_of_t_horizon(k=1),
        )


def test_rejects_multiple_k_out_of_t_horizons():
    process = _k_out_of_t_horizon() | _k_out_of_t_horizon()

    with pytest.raises(CheckpointError, match="exactly one"):
        validate_k_out_of_t_checkpoint(
            sampler_state=_k_out_of_t_state(),
            current_law=KOutOfTSamplingLaw(2, 8, "block", 100),
            accounted_process=process,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_rate", "0.1"),
        ("sample_rate", True),
        ("truncated_batch_size", 10.5),
        ("num_samples", "100"),
    ],
)
def test_rejects_coercible_poisson_sampler_fields(field, value):
    state = _sampler_state(cap=10)
    state[field] = value

    with pytest.raises(CheckpointError, match=field):
        validate_poisson_checkpoint(
            sampler_state=state,
            runtime_sample_rate=0.1,
            current_law=PoissonSamplingLaw(0.1, 10, 100),
            accounted_process=_step(cap=10) * 4,
            global_step=4,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("k", True),
        ("t", 8.5),
        ("allocation", 1),
        ("num_samples", "100"),
    ],
)
def test_rejects_coercible_k_out_of_t_sampler_fields(field, value):
    state = _k_out_of_t_state()
    state[field] = value

    with pytest.raises(CheckpointError, match=field):
        validate_k_out_of_t_checkpoint(
            sampler_state=state,
            current_law=KOutOfTSamplingLaw(2, 8, "block", 100),
            accounted_process=_k_out_of_t_horizon(),
        )
