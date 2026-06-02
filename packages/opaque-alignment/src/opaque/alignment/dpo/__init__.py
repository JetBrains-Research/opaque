"""opaque.alignment.dpo — functional DPO primitives (method namespace).

Mirrors ``opaque.dpsgd`` / ``opaque.dpftrl``: the method's primitives live in
sub-concern subpackages, and a small curated headline re-exports the primary
workflow entry points (data prep → collation → reference precompute →
telemetry). The per-pair heads live in :mod:`opaque.alignment.dpo.loss`; the
log-prob primitives they consume — ``sequence_logp`` and its memory-efficient
drop-in ``fused_sequence_logp`` — live in the sibling
:mod:`opaque.alignment.dpo.logp` (a logp is not a loss). A DPO
``per_example_loss`` is ``head(sequence_logp(...) − ref_logp, …)``.
"""

from opaque.alignment.dpo import collator, data, logp, loss, metric, reference
from opaque.alignment.dpo.collator import preference_collator
from opaque.alignment.dpo.data import extract_prompt
from opaque.alignment.dpo.metric import reward_metrics
from opaque.alignment.dpo.reference import compute_ref_logprobs_for_dataset

__all__ = [
    # sub-concern subpackages
    "loss",
    "logp",
    "collator",
    "reference",
    "metric",
    "data",
    # curated workflow headline (individual losses live in ``dpo.loss``)
    "extract_prompt",
    "preference_collator",
    "compute_ref_logprobs_for_dataset",
    "reward_metrics",
]
