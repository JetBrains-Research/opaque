"""opaque.alignment.loss.kto façade — re-exports the KTO family.

Registry + DP-purity records live in :mod:`opaque.alignment.loss.kto.types`.
"""

from opaque.api.alignment.loss.kto import (
    KTO_AGGREGATES,
    KTO_LOSSES,
    KTO_SPEC,
    KtoVariant,
    apo_zero_unpaired,
    kto_loss,
    resolve_kto_loss,
)

__all__ = [
    "KTO_LOSSES",
    "KTO_SPEC",
    "KTO_AGGREGATES",
    "KtoVariant",
    "resolve_kto_loss",
    "kto_loss",
    "apo_zero_unpaired",
]
