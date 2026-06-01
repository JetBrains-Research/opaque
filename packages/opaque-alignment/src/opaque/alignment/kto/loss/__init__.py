"""opaque.alignment.kto.loss façade — re-exports the KTO loss family.

``kto_loss`` (Tier 2 — detached ``kl_mean`` aggregate) and ``apo_zero_unpaired``
(Tier 1). The registry / DP-purity records (``KTO_LOSSES``, ``KTO_SPEC``,
``KTO_AGGREGATES``) are kept because they are load-bearing: ``KTO_AGGREGATES``
declares the Tier-2 aggregate the trainer must compute, and ``resolve_kto_loss``
is the config-string boundary for the kernel dispatcher.
"""

from opaque.api.alignment.kto.loss import (
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
