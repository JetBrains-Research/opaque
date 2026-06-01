"""opaque.alignment.loss.sft façade — re-exports the SFT family.

Registry + DP-purity records live in :mod:`opaque.alignment.loss.sft.types`.
"""

from opaque.api.alignment.loss.sft import (
    SFT_LOSSES,
    SFT_SPEC,
    SftVariant,
    dft_loss,
    nll_loss,
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
