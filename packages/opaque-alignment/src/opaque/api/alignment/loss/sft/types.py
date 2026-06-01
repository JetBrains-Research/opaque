"""SFT variant enum, ``SFT_LOSSES`` registry, and DP-purity declarations
(plan §7.3, §8.2).

``chunked_nll`` is a documented alias of ``nll`` — it is math-equivalent; the
chunked kernel selection (memory optimization) lives in ``kernel/`` (§7.10),
not in the loss math. All SFT variants are Tier 1 (per-example divisor).
"""

from __future__ import annotations

from typing import Callable, Literal

import torch

from opaque.api.alignment.loss.sft._dft import dft_loss
from opaque.api.alignment.loss.sft._nll import nll_loss
from opaque.api.alignment.loss.types import DPSpec

LossFn = Callable[..., torch.Tensor]

SftVariant = Literal["nll", "dft", "chunked_nll"]

SFT_LOSSES: dict[str, LossFn] = {
    "nll": nll_loss,
    "dft": dft_loss,
    "chunked_nll": nll_loss,  # math-equivalent alias of nll (kernel-level chunking)
}

SFT_SPEC: dict[str, DPSpec] = {
    "nll": DPSpec(tier=1),
    "dft": DPSpec(tier=1),
    "chunked_nll": DPSpec(tier=1),
}


def resolve_sft_loss(loss_type: str) -> LossFn:
    """Return the SFT loss callable for ``loss_type`` (raises ``KeyError`` if
    unknown). ``chunked_nll`` resolves to ``nll`` (math-equivalent)."""
    try:
        return SFT_LOSSES[loss_type]
    except KeyError as exc:
        raise KeyError(
            f"Unknown SFT loss_type {loss_type!r}. Available: {sorted(SFT_LOSSES)}"
        ) from exc


__all__ = ["SftVariant", "SFT_LOSSES", "SFT_SPEC", "resolve_sft_loss"]
