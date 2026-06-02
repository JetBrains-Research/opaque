"""opaque.alignment.dpo.loss façade — re-exports the DPO loss family.

Direct functions only (mirrors ``opaque.alignment.sft.loss``): the 14 per-pair
heads on log-ratios and the ``f_divergence_*`` / ``mpo_combine`` /
``wpo_weights`` / ``ld_dpo_split`` log-ratio helpers. String dispatch is the
caller's concern.

Naming scheme: ``<method>[_<variant>]_loss``. The pairwise/preference context is
implied by this namespace and never spelled out (so ``exo_loss``, not
``exo_pair_loss``); a ``_<variant>`` qualifier appears only to disambiguate
*co-existing* variants of one method — ``sigmoid_loss`` / ``sigmoid_norm_loss``
and ``apo_zero_loss`` / ``apo_down_loss``. ``chosen_nll_loss`` is the
chosen-completion NLL regulariser used in MPO/RPO blends (it is *not* the SFT
method's loss — that lives in ``opaque.alignment.sft.loss``).

The fused DPO path is *not* a loss here: it is the memory-efficient
``fused_sequence_logp`` in the sibling :mod:`opaque.alignment.dpo.logp`, composed
with these per-pair heads exactly as the eager ``sequence_logp`` is — the kernel
produces logp, the head stays one of these functions.
"""

from opaque.api.alignment.dpo.loss import (
    apo_down_loss,
    apo_zero_loss,
    bco_loss,
    discopop_loss,
    exo_loss,
    hinge_loss,
    ipo_loss,
    nca_loss,
    robust_loss,
    chosen_nll_loss,
    sigmoid_loss,
    sigmoid_norm_loss,
    sppo_loss,
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
    "chosen_nll_loss",
    "squarechipo_loss",
    "apo_zero_loss",
    "apo_down_loss",
    "exo_loss",
    "nca_loss",
    "bco_loss",
    "sppo_loss",
    # log-ratio helpers
    "f_divergence_remap",
    "f_divergence_logits",
    "mpo_combine",
    "wpo_weights",
    "ld_dpo_split",
]
