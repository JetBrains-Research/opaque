"""KTO registry + Tier-2 declarations (plan §7.2, §8.3, §9)."""

from __future__ import annotations

import typing

import pytest

from opaque.api.alignment.kto.loss.types import (
    KTO_AGGREGATES,
    KTO_LOSSES,
    KTO_SPEC,
    KtoVariant,
    resolve_kto_loss,
)


def test_registry_keys() -> None:
    assert set(KTO_LOSSES) == {"kto", "apo_zero_unpaired"}
    assert set(typing.get_args(KtoVariant)) == set(KTO_LOSSES)


def test_kto_is_tier2_with_detached_kl_mean_aggregate() -> None:
    spec = KTO_SPEC["kto"]
    assert spec.tier == 2
    assert spec.cross_batch_aggregate == "kl_mean"
    assert spec.aggregate_must_detach is True
    assert spec.aggregate_leverage == "O(1/n)"

    agg = KTO_AGGREGATES["kto"]
    assert agg.name == "kl_mean"
    assert agg.detach is True
    assert agg.cross_rank is False  # v1: per-rank batch-mean KL (§9.4)


def test_apo_zero_unpaired_is_tier1() -> None:
    assert KTO_SPEC["apo_zero_unpaired"].tier == 1


def test_resolve_and_unknown() -> None:
    assert resolve_kto_loss("kto") is KTO_LOSSES["kto"]
    with pytest.raises(KeyError, match="Unknown KTO loss_type"):
        resolve_kto_loss("nope")


def test_facade_and_top_level() -> None:
    import opaque.alignment as top
    from opaque.alignment.kto.loss import KTO_LOSSES as facade

    assert top.KTO_LOSSES is facade
