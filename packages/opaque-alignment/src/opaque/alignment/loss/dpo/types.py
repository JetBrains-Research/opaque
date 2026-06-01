"""opaque.alignment.loss.dpo types façade — registry + DP-purity records."""

from opaque.api.alignment.loss.dpo.types import (
    DPO_LOSSES,
    DPO_SPEC,
    DpoVariant,
    resolve_dpo_loss,
)

__all__ = ["DpoVariant", "DPO_LOSSES", "DPO_SPEC", "resolve_dpo_loss"]
