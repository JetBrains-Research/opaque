"""DPO loss family impl — per-pair scalar losses and log-ratio helpers.

All variants are pure, elementwise, Tier-1 per-example functions (plan §7.1):
swapping one example's data changes only that example's gradient, enforced by
the ``vmap(grad(...))`` NaN-injection sweep in
``tests/dpo/loss/test_vmap_safety.py`` rather than carried as loss metadata.

Direct functions only — there is no string registry / resolver / variant enum
(mirrors ``opaque.api.alignment.sft.loss``). A name→function dispatch is the
caller's concern: a config-string consumer (trainer / CLI) builds its own small
mapping at the call site, as ``examples/train_dpo.py`` does. The
``f_divergence_*``, ``mpo_combine``, ``wpo_weights``, and ``ld_dpo_split``
helpers preprocess log-ratios before a variant.
"""

from opaque.api.alignment.dpo.loss._apo import dpo_apo_down, dpo_apo_zero
from opaque.api.alignment.dpo.loss._bco import dpo_bco_pair
from opaque.api.alignment.dpo.loss._discopop import dpo_discopop
from opaque.api.alignment.dpo.loss._exo import dpo_exo_pair
from opaque.api.alignment.dpo.loss._f_divergence import (
    f_divergence_logits,
    f_divergence_remap,
)
from opaque.api.alignment.dpo.loss._hinge import dpo_hinge
from opaque.api.alignment.dpo.loss._ipo import dpo_ipo
from opaque.api.alignment.dpo.loss._ld_dpo import ld_dpo_split
from opaque.api.alignment.dpo.loss._mpo import mpo_combine
from opaque.api.alignment.dpo.loss._nca import dpo_nca_pair
from opaque.api.alignment.dpo.loss._robust import dpo_robust
from opaque.api.alignment.dpo.loss._sft import dpo_sft
from opaque.api.alignment.dpo.loss._sigmoid import dpo_sigmoid
from opaque.api.alignment.dpo.loss._sigmoid_norm import dpo_sigmoid_norm
from opaque.api.alignment.dpo.loss._sppo import dpo_sppo_hard
from opaque.api.alignment.dpo.loss._squarechipo import dpo_squarechipo
from opaque.api.alignment.dpo.loss._wpo import wpo_weights

__all__ = [
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
