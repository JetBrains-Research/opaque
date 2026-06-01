"""opaque.alignment.metric façade — re-exports alignment metrics."""

from opaque.api.alignment.metric import (
    entropy_from_logits,
    kl_estimator,
    mean_token_accuracy,
    reward_metrics,
)

__all__ = [
    "reward_metrics",
    "kl_estimator",
    "entropy_from_logits",
    "mean_token_accuracy",
]
