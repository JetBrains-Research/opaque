"""DPO loss family impl — per-pair scalar losses, helpers, and registry.

All variants are pure, elementwise, Tier-1 per-example functions (plan §7.1).
Individual functions are exported for direct use in functional examples; the
``DPO_LOSSES`` registry + ``resolve_dpo_loss`` provide string dispatch for
trainers/configs. The ``f_divergence_*``, ``mpo_combine``, ``wpo_weights``,
and ``ld_dpo_split`` helpers preprocess log-ratios before a variant.
"""

from opaque.api.alignment.loss.dpo._apo import dpo_apo_down, dpo_apo_zero
from opaque.api.alignment.loss.dpo._bco import dpo_bco_pair
from opaque.api.alignment.loss.dpo._discopop import dpo_discopop
from opaque.api.alignment.loss.dpo._exo import dpo_exo_pair
from opaque.api.alignment.loss.dpo._f_divergence import (
    f_divergence_logits,
    f_divergence_remap,
)
from opaque.api.alignment.loss.dpo._hinge import dpo_hinge
from opaque.api.alignment.loss.dpo._ipo import dpo_ipo
from opaque.api.alignment.loss.dpo._ld_dpo import ld_dpo_split
from opaque.api.alignment.loss.dpo._mpo import mpo_combine
from opaque.api.alignment.loss.dpo._nca import dpo_nca_pair
from opaque.api.alignment.loss.dpo._robust import dpo_robust
from opaque.api.alignment.loss.dpo._sft import dpo_sft
from opaque.api.alignment.loss.dpo._sigmoid import dpo_sigmoid
from opaque.api.alignment.loss.dpo._sigmoid_norm import dpo_sigmoid_norm
from opaque.api.alignment.loss.dpo._sppo import dpo_sppo_hard
from opaque.api.alignment.loss.dpo._squarechipo import dpo_squarechipo
from opaque.api.alignment.loss.dpo._wpo import wpo_weights
from opaque.api.alignment.loss.dpo.types import (
    DPO_LOSSES,
    DPO_SPEC,
    DpoVariant,
    resolve_dpo_loss,
)

__all__ = [
    # registry + dispatch
    "DPO_LOSSES",
    "DPO_SPEC",
    "DpoVariant",
    "resolve_dpo_loss",
    # variants
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
    # helpers
    "f_divergence_remap",
    "f_divergence_logits",
    "mpo_combine",
    "wpo_weights",
    "ld_dpo_split",
]
