"""opaque.alignment.dpo — functional DPO primitives (method namespace).

Mirrors ``opaque.dpsgd`` / ``opaque.dpftrl``: the method's primitives live in
sub-concern subpackages, and a small curated headline re-exports the primary
workflow entry points (data prep → collation → reference precompute →
telemetry). The 14 per-pair losses live in :mod:`opaque.alignment.dpo.loss`;
the opt-in fused kernel in :mod:`opaque.alignment.dpo.kernel`. Lower-level
shared primitives (``sequence_logp`` etc.) stay internal under
``opaque.api.alignment`` for power users.
"""

from opaque.alignment.dpo import collator, data, kernel, loss, metric, reference
from opaque.alignment.dpo.collator import preference_collator
from opaque.alignment.dpo.data import extract_prompt
from opaque.alignment.dpo.metric import reward_metrics
from opaque.alignment.dpo.reference import compute_ref_logprobs_for_dataset

__all__ = [
    # sub-concern subpackages
    "loss",
    "collator",
    "kernel",
    "reference",
    "metric",
    "data",
    # curated workflow headline (individual losses live in ``dpo.loss``)
    "extract_prompt",
    "preference_collator",
    "compute_ref_logprobs_for_dataset",
    "reward_metrics",
]
