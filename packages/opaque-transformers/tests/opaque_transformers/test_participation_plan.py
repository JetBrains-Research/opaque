"""Behavioral tests for the Trainer's immutable participation contract."""

from dataclasses import FrozenInstanceError

import pytest

from opaque.api.transformers.trainer._participation import (
    ALLOWED_SAMPLERS,
    ResolvedParticipationPlan,
)
from opaque.exceptions import CheckpointError, ConfigurationError


def _resolve(**overrides):
    values = {
        "mechanism_kind": "gaussian",
        "sampling_mode": "poisson",
        "sampling_kwargs": {},
        "population_size": 100,
        "expected_batch_size": 10,
        "sample_rate": 0.1,
        "total_steps": 20,
        "num_bins": 10,
        "world_size": 1,
    }
    values.update(overrides)
    return ResolvedParticipationPlan.resolve(**values)


def test_plan_is_deeply_immutable_and_canonicalizes_poisson_alias() -> None:
    original = {"max_batch_size": 8}
    plan = _resolve(sampling_kwargs=original)
    original["max_batch_size"] = 1

    assert plan.sampling_kwargs == (("truncated_batch_size", 8),)
    with pytest.raises(FrozenInstanceError):
        plan.sampling_mode = "k_out_of_t"


def test_plan_state_round_trip_is_canonical_and_independent() -> None:
    plan = _resolve(
        sampling_mode="k_out_of_t",
        sampling_kwargs={"k": 2, "allocation": "block"},
    )
    saved = plan.to_state_dict()
    restored = ResolvedParticipationPlan.from_state_dict(saved)
    saved["sampling_kwargs"]["k"] = 9

    assert restored == plan
    assert restored.sampler_kwargs_dict() == {"allocation": "block", "k": 2}


@pytest.mark.parametrize("mode", ["cyclic_poisson", "sequential"])
def test_plan_rejects_transformer_modes_without_accountants(mode) -> None:
    with pytest.raises(ConfigurationError, match="not valid"):
        _resolve(sampling_mode=mode)


def test_plan_rejects_band_mf_with_plain_poisson() -> None:
    with pytest.raises(
        ConfigurationError,
        match=r"plain whole-dataset Poisson.*b_min_sep.*mf_identity",
    ):
        _resolve(mechanism_kind="mf_band")


def test_checkpoint_plan_rejects_malformed_schema() -> None:
    saved = _resolve().to_state_dict()
    saved["unexpected"] = True
    with pytest.raises(CheckpointError, match="fields do not match"):
        ResolvedParticipationPlan.from_state_dict(saved)


def test_resolve_rejects_mutated_non_mapping_kwargs() -> None:
    with pytest.raises(ConfigurationError, match="must be a mapping"):
        _resolve(sampling_kwargs=[("truncated_batch_size", 8)])


@pytest.mark.parametrize(
    "allocation",
    [
        type("Allocation", (str,), {})("block"),
        type("EqualToBlock", (), {"__eq__": lambda self, other: other == "block"})(),
    ],
)
def test_plan_rejects_non_builtin_allocation_strings(allocation) -> None:
    with pytest.raises(ConfigurationError, match=r"allocation.*block.*total"):
        _resolve(
            sampling_mode="k_out_of_t",
            sampling_kwargs={"k": 2, "allocation": allocation},
        )


def test_plan_exposes_rank_local_population() -> None:
    plan = _resolve(
        population_size=96,
        expected_batch_size=12,
        sample_rate=0.125,
        world_size=3,
    )

    assert plan.local_population_size == 32


@pytest.mark.parametrize(
    ("mechanism", "sampling_mode", "sampling_kwargs", "expected"),
    [
        ("gaussian", "poisson", {}, False),
        ("gaussian", "k_out_of_t", {"k": 2, "allocation": "block"}, True),
        ("mf_identity", "poisson", {}, True),
        ("mf_band", "b_min_sep", {}, True),
    ],
)
def test_plan_owns_accounting_lifecycle(
    mechanism,
    sampling_mode,
    sampling_kwargs,
    expected,
) -> None:
    plan = _resolve(
        mechanism_kind=mechanism,
        sampling_mode=sampling_mode,
        sampling_kwargs=sampling_kwargs,
    )

    assert plan.requires_horizon_process is expected


def test_plan_rejects_rank_local_truncation_under_ddp() -> None:
    with pytest.raises(
        ConfigurationError,
        match="truncated Poisson sampling is not supported with distributed",
    ):
        _resolve(
            sampling_kwargs={"truncated_batch_size": 8},
            population_size=96,
            expected_batch_size=12,
            sample_rate=0.125,
            world_size=3,
        )


def test_mechanism_sampler_registry_is_immutable() -> None:
    with pytest.raises(TypeError):
        ALLOWED_SAMPLERS["mf_band"] = frozenset({"poisson"})
