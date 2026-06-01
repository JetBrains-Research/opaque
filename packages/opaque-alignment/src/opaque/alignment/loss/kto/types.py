"""opaque.alignment.loss.kto types façade — registry + DP-purity records."""

from opaque.api.alignment.loss.kto.types import (
    KTO_AGGREGATES,
    KTO_LOSSES,
    KTO_SPEC,
    KtoVariant,
    resolve_kto_loss,
)

__all__ = ["KtoVariant", "KTO_LOSSES", "KTO_SPEC", "KTO_AGGREGATES", "resolve_kto_loss"]
