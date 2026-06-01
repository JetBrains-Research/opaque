"""SFT registry + chunked_nll alias (plan §7.3, §8.2)."""

from __future__ import annotations

import typing

import pytest

from opaque.api.alignment.loss.sft.types import (
    SFT_LOSSES,
    SFT_SPEC,
    SftVariant,
    resolve_sft_loss,
)


def test_registry_keys() -> None:
    assert set(SFT_LOSSES) == {"nll", "dft", "chunked_nll"}
    assert set(typing.get_args(SftVariant)) == set(SFT_LOSSES)


def test_chunked_nll_is_alias_of_nll() -> None:
    assert SFT_LOSSES["chunked_nll"] is SFT_LOSSES["nll"]
    assert resolve_sft_loss("chunked_nll") is resolve_sft_loss("nll")


def test_all_tier1() -> None:
    for name in SFT_LOSSES:
        assert SFT_SPEC[name].tier == 1


def test_resolve_and_unknown() -> None:
    assert resolve_sft_loss("dft") is SFT_LOSSES["dft"]
    with pytest.raises(KeyError, match="Unknown SFT loss_type"):
        resolve_sft_loss("nope")


def test_facade_and_top_level() -> None:
    import opaque.alignment as top
    from opaque.alignment.loss.sft import SFT_LOSSES as facade

    assert top.SFT_LOSSES is facade
