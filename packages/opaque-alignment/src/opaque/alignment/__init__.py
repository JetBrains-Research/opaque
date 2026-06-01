"""opaque.alignment — functional primitives for DP-safe preference learning.

Headline re-exports. The loss-family registries (``DPO_LOSSES``,
``KTO_LOSSES``, ``SFT_LOSSES``), reference helpers, and packing transforms
land as their phases ship; this façade grows incrementally.

See ``docs/development/opaque-alignment-plan.md`` for the package design.
"""

from opaque.alignment.collator import (
    language_modeling_collator,
    preference_collator,
    unpaired_preference_collator,
)
from opaque.alignment.data import extract_prompt, rotate_kto_completions
from opaque.alignment.logprob import (
    get_batch_logps,
    selective_log_softmax,
    sequence_logp,
)
from opaque.alignment.loss.dpo import DPO_LOSSES
from opaque.alignment.loss.kto import KTO_LOSSES
from opaque.alignment.loss.sft import SFT_LOSSES
from opaque.alignment.loss.types import DPSpec, LossAggregateSpec
from opaque.alignment.metric import (
    entropy_from_logits,
    kl_estimator,
    mean_token_accuracy,
    reward_metrics,
)
from opaque.alignment.reference import (
    compute_ref_logprobs_for_dataset,
    ema_update_reference,
    null_ref_context,
)

__all__ = [
    "sequence_logp",
    "selective_log_softmax",
    "get_batch_logps",
    "DPO_LOSSES",
    "KTO_LOSSES",
    "SFT_LOSSES",
    "DPSpec",
    "LossAggregateSpec",
    "language_modeling_collator",
    "preference_collator",
    "unpaired_preference_collator",
    "extract_prompt",
    "rotate_kto_completions",
    "compute_ref_logprobs_for_dataset",
    "null_ref_context",
    "ema_update_reference",
    "reward_metrics",
    "kl_estimator",
    "entropy_from_logits",
    "mean_token_accuracy",
]
