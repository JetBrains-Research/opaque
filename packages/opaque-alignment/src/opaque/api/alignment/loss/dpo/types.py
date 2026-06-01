"""DPO variant enum, ``DPO_LOSSES`` registry, and ``DPO_SPEC`` DP-purity
declarations (plan §7.1, §8.3).

The registry is the single source of truth for which DPO variants exist. All
shipped variants are Tier 1 (strict per-example). The ``aot`` family is
rejected at this layer (Tier 3 — sort-across-batch breaks per-example
sensitivity); ``resolve_dpo_loss`` raises for it (plan §3.3, §11.7).
"""

from __future__ import annotations

from typing import Callable, Literal

import torch

from opaque.api.alignment.loss.dpo._apo import dpo_apo_down, dpo_apo_zero
from opaque.api.alignment.loss.dpo._bco import dpo_bco_pair
from opaque.api.alignment.loss.dpo._discopop import dpo_discopop
from opaque.api.alignment.loss.dpo._exo import dpo_exo_pair
from opaque.api.alignment.loss.dpo._hinge import dpo_hinge
from opaque.api.alignment.loss.dpo._ipo import dpo_ipo
from opaque.api.alignment.loss.dpo._nca import dpo_nca_pair
from opaque.api.alignment.loss.dpo._robust import dpo_robust
from opaque.api.alignment.loss.dpo._sft import dpo_sft
from opaque.api.alignment.loss.dpo._sigmoid import dpo_sigmoid
from opaque.api.alignment.loss.dpo._sigmoid_norm import dpo_sigmoid_norm
from opaque.api.alignment.loss.dpo._sppo import dpo_sppo_hard
from opaque.api.alignment.loss.dpo._squarechipo import dpo_squarechipo
from opaque.api.alignment.loss.types import DPSpec

LossFn = Callable[..., torch.Tensor]

DpoVariant = Literal[
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
]

DPO_LOSSES: dict[str, LossFn] = {
    "sigmoid": dpo_sigmoid,
    "hinge": dpo_hinge,
    "ipo": dpo_ipo,
    "robust": dpo_robust,
    "apo_zero": dpo_apo_zero,
    "apo_down": dpo_apo_down,
    "exo_pair": dpo_exo_pair,
    "nca_pair": dpo_nca_pair,
    "bco_pair": dpo_bco_pair,
    "sppo_hard": dpo_sppo_hard,
    "discopop": dpo_discopop,
    "sft": dpo_sft,
    "sigmoid_norm": dpo_sigmoid_norm,
    "squarechipo": dpo_squarechipo,
}

_AOT_REJECTION = (
    "sort-across-batch with O(1) leverage breaks per-example DP sensitivity; "
    "no published DP-safe variant exists (plan §3.3 Tier 3)."
)

# Every shipped variant is Tier 1. The aot* family is recorded as Tier 3 /
# not DP-safe so the rejection rationale is discoverable, but it is NOT a key
# of DPO_LOSSES — there is no exposure path to a Tier-3 loss callable.
DPO_SPEC: dict[str, DPSpec] = {name: DPSpec(tier=1) for name in DPO_LOSSES}
DPO_SPEC.update(
    {
        "aot": DPSpec(
            tier=3,
            dp_safe=False,
            aggregate_leverage="sort",
            rejection_reason=_AOT_REJECTION,
        ),
        "aot_pair": DPSpec(
            tier=3,
            dp_safe=False,
            aggregate_leverage="sort",
            rejection_reason=_AOT_REJECTION,
        ),
        "aot_unpaired": DPSpec(
            tier=3,
            dp_safe=False,
            aggregate_leverage="sort",
            rejection_reason=_AOT_REJECTION,
        ),
    }
)


def resolve_dpo_loss(loss_type: str) -> LossFn:
    """Return the loss callable for ``loss_type``.

    Tier-3 (rejected) variants raise :class:`NotImplementedError` carrying the
    rejection rationale; unknown names raise :class:`KeyError` (plan §11.7).
    """
    spec = DPO_SPEC.get(loss_type)
    if spec is not None and spec.tier == 3:
        raise NotImplementedError(
            f"DPO loss_type {loss_type!r} is rejected for DP training: "
            f"{spec.rejection_reason}"
        )
    try:
        return DPO_LOSSES[loss_type]
    except KeyError as exc:
        raise KeyError(
            f"Unknown DPO loss_type {loss_type!r}. Available: {sorted(DPO_LOSSES)}"
        ) from exc


__all__ = ["DpoVariant", "DPO_LOSSES", "DPO_SPEC", "resolve_dpo_loss"]
