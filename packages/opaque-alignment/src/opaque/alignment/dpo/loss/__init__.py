"""opaque.alignment.dpo.loss façade — re-exports the DPO loss family.

Direct functions only (mirrors ``opaque.alignment.sft.loss``): the 14 per-pair
variants plus the ``f_divergence_*`` / ``mpo_combine`` / ``wpo_weights`` /
``ld_dpo_split`` log-ratio helpers. String dispatch is the caller's concern.
"""

from opaque.api.alignment.dpo.loss import (
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

__all__ = [
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
    "f_divergence_remap",
    "f_divergence_logits",
    "mpo_combine",
    "wpo_weights",
    "ld_dpo_split",
]
