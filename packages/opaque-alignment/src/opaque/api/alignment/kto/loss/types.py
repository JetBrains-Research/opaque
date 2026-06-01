"""KTO variant enum, ``KTO_LOSSES`` registry, and DP-purity declarations
(plan §7.2, §8.3, §9).

``kto`` is **Tier 2** — it consumes a detached batch-mean KL aggregate
(``LossAggregateSpec("kl_mean")``) with ``O(1/n)`` leverage; the caller must
compute + detach it outside the vmap region (§8.1). ``apo_zero_unpaired`` is
Tier 1.
"""

from __future__ import annotations

from typing import Callable, Literal

import torch

from opaque.api.alignment.kto.loss._apo_zero_unpaired import apo_zero_unpaired
from opaque.api.alignment.kto.loss._kto import kto_loss
from opaque.api.alignment.loss.types import DPSpec, LossAggregateSpec

LossFn = Callable[..., torch.Tensor]

KtoVariant = Literal["kto", "apo_zero_unpaired"]

KTO_LOSSES: dict[str, LossFn] = {
    "kto": kto_loss,
    "apo_zero_unpaired": apo_zero_unpaired,
}

KTO_SPEC: dict[str, DPSpec] = {
    "kto": DPSpec(
        tier=2,
        cross_batch_aggregate="kl_mean",
        aggregate_must_detach=True,
        aggregate_leverage="O(1/n)",
    ),
    "apo_zero_unpaired": DPSpec(tier=1),
}

# Tier-2 aggregate declarations: the trainer reads these to know which
# aggregate to compute pre-vmap (and, when cross_rank=True in v2, to all-reduce
# — see plan §9). v1 uses a per-rank batch-mean KL (cross_rank=False).
KTO_AGGREGATES: dict[str, LossAggregateSpec] = {
    "kto": LossAggregateSpec(
        name="kl_mean", reduction="mean", detach=True, cross_rank=False
    ),
}


def resolve_kto_loss(loss_type: str) -> LossFn:
    """Return the KTO loss callable for ``loss_type`` (raises ``KeyError`` if
    unknown). All KTO variants are DP-safe (Tier 1/2); none are rejected."""
    try:
        return KTO_LOSSES[loss_type]
    except KeyError as exc:
        raise KeyError(
            f"Unknown KTO loss_type {loss_type!r}. Available: {sorted(KTO_LOSSES)}"
        ) from exc


__all__ = ["KtoVariant", "KTO_LOSSES", "KTO_SPEC", "KTO_AGGREGATES", "resolve_kto_loss"]
