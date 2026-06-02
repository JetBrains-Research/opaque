"""opaque.alignment.dpo.loss façade — re-exports the DPO loss family.

Direct functions only (mirrors ``opaque.alignment.sft.loss``): the 14 per-pair
variants on log-ratios, the ``f_divergence_*`` / ``mpo_combine`` /
``wpo_weights`` / ``ld_dpo_split`` log-ratio helpers, and the memory-efficient
``fused_linear_dpo_loss`` (fuses the lm-head projection + per-sequence logp +
per-pair loss; auto-selects a fused kernel vs pure-torch). String dispatch is
the caller's concern.
"""

from opaque.api.alignment.dpo.kernel import fused_linear_dpo_loss
from opaque.api.alignment.dpo.loss import (
    apo_down_loss,
    apo_zero_loss,
    bco_pair_loss,
    discopop_loss,
    exo_pair_loss,
    hinge_loss,
    ipo_loss,
    nca_pair_loss,
    robust_loss,
    sft_loss,
    sigmoid_loss,
    sigmoid_norm_loss,
    sppo_hard_loss,
    squarechipo_loss,
    f_divergence_logits,
    f_divergence_remap,
    ld_dpo_split,
    mpo_combine,
    wpo_weights,
)

__all__ = [
    # per-pair variants (on log-ratios)
    "sigmoid_loss",
    "hinge_loss",
    "robust_loss",
    "ipo_loss",
    "sigmoid_norm_loss",
    "discopop_loss",
    "sft_loss",
    "squarechipo_loss",
    "apo_zero_loss",
    "apo_down_loss",
    "exo_pair_loss",
    "nca_pair_loss",
    "bco_pair_loss",
    "sppo_hard_loss",
    # memory-efficient fused loss (on hidden states)
    "fused_linear_dpo_loss",
    # log-ratio helpers
    "f_divergence_remap",
    "f_divergence_logits",
    "mpo_combine",
    "wpo_weights",
    "ld_dpo_split",
]
