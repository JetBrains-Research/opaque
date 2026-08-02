"""opaque.alignment.metric — shared token-level metrics.

Detached, un-noised token telemetry; release is outside Opaque's DP accounting.
Preference metrics live in :mod:`opaque.alignment.dpo.metric`.
"""

from opaque.api.alignment.metric import entropy_from_logits, mean_token_accuracy

__all__ = ["entropy_from_logits", "mean_token_accuracy"]
