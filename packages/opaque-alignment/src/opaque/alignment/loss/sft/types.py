"""opaque.alignment.loss.sft types façade — registry + DP-purity records."""

from opaque.api.alignment.loss.sft.types import (
    SFT_LOSSES,
    SFT_SPEC,
    SftVariant,
    resolve_sft_loss,
)

__all__ = ["SftVariant", "SFT_LOSSES", "SFT_SPEC", "resolve_sft_loss"]
