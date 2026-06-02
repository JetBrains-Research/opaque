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

from opaque.api.alignment.dpo.loss._apo import apo_down_loss, apo_zero_loss
from opaque.api.alignment.dpo.loss._bco import bco_pair_loss
from opaque.api.alignment.dpo.loss._discopop import discopop_loss
from opaque.api.alignment.dpo.loss._exo import exo_pair_loss
from opaque.api.alignment.dpo.loss._f_divergence import (
    f_divergence_logits,
    f_divergence_remap,
)
from opaque.api.alignment.dpo.loss._hinge import hinge_loss
from opaque.api.alignment.dpo.loss._ipo import ipo_loss
from opaque.api.alignment.dpo.loss._ld_dpo import ld_dpo_split
from opaque.api.alignment.dpo.loss._mpo import mpo_combine
from opaque.api.alignment.dpo.loss._nca import nca_pair_loss
from opaque.api.alignment.dpo.loss._robust import robust_loss
from opaque.api.alignment.dpo.loss._sft import sft_loss
from opaque.api.alignment.dpo.loss._sigmoid import sigmoid_loss
from opaque.api.alignment.dpo.loss._sigmoid_norm import sigmoid_norm_loss
from opaque.api.alignment.dpo.loss._sppo import sppo_hard_loss
from opaque.api.alignment.dpo.loss._squarechipo import squarechipo_loss
from opaque.api.alignment.dpo.loss._wpo import wpo_weights

__all__ = [
    # variants
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
    # helpers
    "f_divergence_remap",
    "f_divergence_logits",
    "mpo_combine",
    "wpo_weights",
    "ld_dpo_split",
]
