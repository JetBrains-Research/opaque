"""opaque.alignment — functional primitives for DP-safe preference learning.

Headline re-exports. The loss-family registries (``DPO_LOSSES``,
``KTO_LOSSES``, ``SFT_LOSSES``), reference helpers, and packing transforms
are added as their phases land; this façade grows incrementally.

See ``docs/development/opaque-alignment-plan.md`` for the package design.
"""

from opaque.alignment.collator import (
    language_modeling_collator,
    preference_collator,
    unpaired_preference_collator,
)
from opaque.alignment.data import extract_prompt
from opaque.alignment.logprob import (
    get_batch_logps,
    selective_log_softmax,
    sequence_logp,
)
from opaque.alignment.loss.types import DPSpec, LossAggregateSpec
from opaque.alignment.metric import (
    entropy_from_logits,
    kl_estimator,
    mean_token_accuracy,
    reward_metrics,
)

__all__ = [
    "sequence_logp",
    "selective_log_softmax",
    "get_batch_logps",
    "DPSpec",
    "LossAggregateSpec",
    "language_modeling_collator",
    "preference_collator",
    "unpaired_preference_collator",
    "extract_prompt",
    "reward_metrics",
    "kl_estimator",
    "entropy_from_logits",
    "mean_token_accuracy",
]
