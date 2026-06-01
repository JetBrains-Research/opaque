"""Unit tests for the shared loss-type records (§7.4).

Cover :class:`DPSpec` and :class:`LossAggregateSpec`: default field values,
explicit (Tier-3 rejection) construction, and the frozen-ness invariant
(mutation raises ``FrozenInstanceError``).
"""

from __future__ import annotations

import dataclasses

import pytest

from opaque.api.alignment.loss.types import DPSpec, LossAggregateSpec


def test_dpspec_defaults() -> None:
    spec = DPSpec(tier=1)
    assert spec.tier == 1
    assert spec.cross_batch_aggregate is None
    assert spec.aggregate_must_detach is True
    assert spec.aggregate_leverage is None
    assert spec.dp_safe is True
    assert spec.rejection_reason is None


def test_dpspec_tier2_aggregate() -> None:
    spec = DPSpec(
        tier=2,
        cross_batch_aggregate="kl_mean",
        aggregate_must_detach=True,
        aggregate_leverage="O(1/n)",
    )
    assert spec.tier == 2
    assert spec.cross_batch_aggregate == "kl_mean"
    assert spec.aggregate_leverage == "O(1/n)"
    assert spec.dp_safe is True


def test_dpspec_tier3_rejected() -> None:
    spec = DPSpec(
        tier=3,
        dp_safe=False,
        rejection_reason="sort-across-batch O(1) leverage",
    )
    assert spec.tier == 3
    assert spec.dp_safe is False
    assert spec.rejection_reason == "sort-across-batch O(1) leverage"


def test_dpspec_frozen() -> None:
    spec = DPSpec(tier=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.tier = 2  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.dp_safe = False  # type: ignore[misc]


def test_loss_aggregate_spec_defaults() -> None:
    spec = LossAggregateSpec(name="kl_mean")
    assert spec.name == "kl_mean"
    assert spec.reduction == "mean"
    assert spec.detach is True
    assert spec.cross_rank is False


def test_loss_aggregate_spec_cross_rank() -> None:
    spec = LossAggregateSpec(name="kl_mean", cross_rank=True)
    assert spec.name == "kl_mean"
    assert spec.cross_rank is True
    # Untouched fields keep their defaults.
    assert spec.reduction == "mean"
    assert spec.detach is True


def test_loss_aggregate_spec_frozen() -> None:
    spec = LossAggregateSpec(name="kl_mean")
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.cross_rank = True  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "other"  # type: ignore[misc]
