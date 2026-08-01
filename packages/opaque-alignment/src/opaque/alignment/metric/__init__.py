"""opaque.alignment.metric — shared token-level metrics.

General token telemetry: mean next-token accuracy and per-token prediction
entropy. Values are detached but un-noised functions of the private batch;
logging or publishing them is outside Opaque's DP accounting. Preference
reward telemetry is DPO-specific and lives in
:mod:`opaque.alignment.dpo.metric`.
"""

from opaque.api.alignment.metric import entropy_from_logits, mean_token_accuracy

__all__ = ["entropy_from_logits", "mean_token_accuracy"]
