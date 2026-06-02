"""opaque.alignment.metric — shared token-level metrics.

General, detached token telemetry for eval logging (not for release): mean
next-token accuracy and per-token prediction entropy. Preference reward
telemetry is DPO-specific and lives in :mod:`opaque.alignment.dpo.metric`.
"""

from opaque.api.alignment.metric import entropy_from_logits, mean_token_accuracy

__all__ = ["entropy_from_logits", "mean_token_accuracy"]
