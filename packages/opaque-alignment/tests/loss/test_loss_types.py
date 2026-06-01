"""Unit tests for the shared loss-type record (§7.4).

Cover :class:`DPSpec`: default field values, explicit (Tier-3 rejection)
construction, and the frozen-ness invariant (mutation raises
``FrozenInstanceError``).
"""

from __future__ import annotations

import dataclasses

import pytest

from opaque.api.alignment.loss.types import DPSpec


def test_dpspec_defaults() -> None:
    spec = DPSpec(tier=1)
    assert spec.tier == 1
    assert spec.dp_safe is True
    assert spec.aggregate_leverage is None
    assert spec.rejection_reason is None


def test_dpspec_tier3_rejected() -> None:
    spec = DPSpec(
        tier=3,
        dp_safe=False,
        aggregate_leverage="sort",
        rejection_reason="sort-across-batch O(1) leverage",
    )
    assert spec.tier == 3
    assert spec.dp_safe is False
    assert spec.aggregate_leverage == "sort"
    assert spec.rejection_reason == "sort-across-batch O(1) leverage"


def test_dpspec_frozen() -> None:
    spec = DPSpec(tier=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.tier = 3  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.dp_safe = False  # type: ignore[misc]
