"""opaque.alignment.dpo — functional DPO primitives (method-first façade).

Mirrors ``opaque.alignment.sft`` / ``opaque.dpsgd``: the DPO method owns its
loss family (:mod:`opaque.alignment.dpo.loss`), preference collator
(:mod:`opaque.alignment.dpo.collator`), fused-linear preference kernel
(:mod:`opaque.alignment.dpo.kernel`), reference-model helpers
(:mod:`opaque.alignment.dpo.reference`), reward telemetry
(:mod:`opaque.alignment.dpo.metric`), and preference prompt extraction
(:mod:`opaque.alignment.dpo.data`).

Shared primitives stay in their concern modules; the ones DPO surfaces are
re-imported here — e.g. ``sequence_logp`` from the shared logprob impl (the
``opaque.dpsgd.clipping`` re-import pattern). Direct loss functions only: there
is no string registry / resolver / variant enum — a config-string consumer
builds its own mapping, as ``examples/train_dpo.py`` does.
"""

from opaque.alignment.dpo.collator import preference_collator
from opaque.alignment.dpo.data import extract_prompt
from opaque.alignment.dpo.kernel import (
    fused_linear_preference,
    opaque_fused_linear_dpo_loss,
)
from opaque.alignment.dpo.loss import (
    dpo_apo_down,
    dpo_apo_zero,
    dpo_bco_pair,
    dpo_discopop,
    dpo_exo_pair,
    dpo_hinge,
    dpo_ipo,
    dpo_nca_pair,
    dpo_robust,
    dpo_sft,
    dpo_sigmoid,
    dpo_sigmoid_norm,
    dpo_sppo_hard,
    dpo_squarechipo,
    f_divergence_logits,
    f_divergence_remap,
    ld_dpo_split,
    mpo_combine,
    wpo_weights,
)
from opaque.alignment.dpo.metric import reward_metrics
from opaque.alignment.dpo.reference import (
    compute_ref_logprobs_for_dataset,
    ema_update_reference,
    null_ref_context,
    with_disabled_adapter,
)

# Shared primitive surfaced through DPO (its public consumer): the per-sequence
# completion log-prob reducer used to form policy log-ratios.
from opaque.api.alignment.logprob import sequence_logp

__all__ = [
    # loss variants
    "dpo_sigmoid",
    "dpo_hinge",
    "dpo_robust",
    "dpo_ipo",
    "dpo_sigmoid_norm",
    "dpo_discopop",
    "dpo_sft",
    "dpo_squarechipo",
    "dpo_apo_zero",
    "dpo_apo_down",
    "dpo_exo_pair",
    "dpo_nca_pair",
    "dpo_bco_pair",
    "dpo_sppo_hard",
    # loss helpers
    "f_divergence_remap",
    "f_divergence_logits",
    "mpo_combine",
    "wpo_weights",
    "ld_dpo_split",
    # collator
    "preference_collator",
    # kernel
    "opaque_fused_linear_dpo_loss",
    "fused_linear_preference",
    # reference
    "compute_ref_logprobs_for_dataset",
    "null_ref_context",
    "with_disabled_adapter",
    "ema_update_reference",
    # metric
    "reward_metrics",
    # data
    "extract_prompt",
    # shared logprob surfaced through DPO
    "sequence_logp",
]
