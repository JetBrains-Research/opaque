"""DPO registry + DP-purity declarations + Tier-3 rejection (plan §8.3, §11.7)."""

from __future__ import annotations

import typing

import pytest

from opaque.api.alignment.loss.dpo.types import (
    DPO_LOSSES,
    DPO_SPEC,
    DpoVariant,
    resolve_dpo_loss,
)

EXPECTED_VARIANTS = {
    "sigmoid",
    "hinge",
    "ipo",
    "robust",
    "apo_zero",
    "apo_down",
    "exo_pair",
    "nca_pair",
    "bco_pair",
    "sppo_hard",
    "discopop",
    "sft",
    "sigmoid_norm",
    "squarechipo",
}


def test_registry_has_exactly_the_expected_variants() -> None:
    assert set(DPO_LOSSES) == EXPECTED_VARIANTS


def test_dpo_variant_literal_matches_registry() -> None:
    literal_args = set(typing.get_args(DpoVariant))
    assert literal_args == EXPECTED_VARIANTS == set(DPO_LOSSES)


def test_every_registry_entry_is_tier_1() -> None:
    for name in DPO_LOSSES:
        assert DPO_SPEC[name].tier == 1
        assert DPO_SPEC[name].dp_safe is True


def test_registry_values_are_callable() -> None:
    for fn in DPO_LOSSES.values():
        assert callable(fn)


def test_resolve_returns_the_callable() -> None:
    assert resolve_dpo_loss("sigmoid") is DPO_LOSSES["sigmoid"]


@pytest.mark.parametrize("rejected", ["aot", "aot_pair", "aot_unpaired"])
def test_tier3_aot_is_rejected(rejected: str) -> None:
    # Tier 3 is recorded in the spec but never exposed as a callable.
    assert rejected not in DPO_LOSSES
    assert DPO_SPEC[rejected].tier == 3
    assert DPO_SPEC[rejected].dp_safe is False
    with pytest.raises(NotImplementedError, match="sort-across-batch"):
        resolve_dpo_loss(rejected)


def test_unknown_loss_type_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="Unknown DPO loss_type"):
        resolve_dpo_loss("does_not_exist")


def test_facade_reexports_registry_and_variants() -> None:
    from opaque.alignment.loss.dpo import DPO_LOSSES as facade_losses
    from opaque.alignment.loss.dpo import dpo_sigmoid

    assert set(facade_losses) == EXPECTED_VARIANTS
    assert facade_losses["sigmoid"] is dpo_sigmoid

    import opaque.alignment as top

    assert top.DPO_LOSSES is facade_losses
