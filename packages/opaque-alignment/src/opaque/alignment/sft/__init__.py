"""opaque.alignment.sft — functional SFT primitives (method namespace).

Mirrors ``opaque.dpsgd`` / ``opaque.dpftrl``: the SFT method owns its loss math
(:mod:`opaque.alignment.sft.loss`) and language-modeling collator
(:mod:`opaque.alignment.sft.collator`), exposed as sub-concern subpackages plus
a small curated headline of the primary entry points.
"""

from opaque.alignment.sft import collator, loss
from opaque.alignment.sft.collator import language_modeling_collator
from opaque.alignment.sft.loss import dft_loss, fused_linear_sft_loss, nll_loss

__all__ = [
    # sub-concern subpackages
    "loss",
    "collator",
    # curated headline
    "nll_loss",
    "dft_loss",
    "fused_linear_sft_loss",
    "language_modeling_collator",
]
