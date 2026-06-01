"""SFT loss family impl — nll, dft (DP-corrected divisor), chunked_nll alias."""

from opaque.api.alignment.loss.sft._dft import dft_loss
from opaque.api.alignment.loss.sft._nll import nll_loss
from opaque.api.alignment.loss.sft.types import (
    SFT_LOSSES,
    SFT_SPEC,
    SftVariant,
    resolve_sft_loss,
)

__all__ = [
    "SFT_LOSSES",
    "SFT_SPEC",
    "SftVariant",
    "resolve_sft_loss",
    "nll_loss",
    "dft_loss",
]
