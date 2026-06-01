"""Alignment metrics impl — rewards, KL estimator, token-level metrics.

Per plan §3.3 telemetry rule, these are private internal state: they return
detached tensors intended for logging/accumulation, not for release.
"""

from opaque.api.alignment.metric._kl import kl_estimator
from opaque.api.alignment.metric._reward import reward_metrics
from opaque.api.alignment.metric._token import entropy_from_logits, mean_token_accuracy

__all__ = [
    "reward_metrics",
    "kl_estimator",
    "entropy_from_logits",
    "mean_token_accuracy",
]
