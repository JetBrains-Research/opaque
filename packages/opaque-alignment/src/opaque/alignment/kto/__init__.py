"""opaque.alignment.kto — functional KTO primitives (method-first façade).

Mirrors the ``opaque.dpsgd`` / ``opaque.dpftrl`` mechanism-namespaced layout.
``kto_loss`` is **Tier 2** — it consumes a detached batch-mean KL aggregate
(``LossAggregateSpec("kl_mean")``, ``O(1/n)`` leverage) the caller computes
outside the vmap region; ``apo_zero_unpaired`` is Tier 1. The method owns its
unpaired-preference collator and the completion rotation that builds the KL
partner; shared primitives (logprob, metric, reference) are reimported from
``opaque.api.alignment.*``.
"""

from opaque.alignment.kto.collator import unpaired_preference_collator
from opaque.alignment.kto.data import rotate_kto_completions
from opaque.alignment.kto.loss import (
    KTO_AGGREGATES,
    KTO_LOSSES,
    KTO_SPEC,
    KtoVariant,
    apo_zero_unpaired,
    kto_loss,
    resolve_kto_loss,
)

__all__ = [
    "kto_loss",
    "apo_zero_unpaired",
    "KTO_LOSSES",
    "KTO_SPEC",
    "KTO_AGGREGATES",
    "KtoVariant",
    "resolve_kto_loss",
    "unpaired_preference_collator",
    "rotate_kto_completions",
]
