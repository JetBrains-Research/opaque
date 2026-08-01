"""Shared alignment metrics impl — exact token-level telemetry.

These functions return detached but un-noised values computed from the private
batch. Detaching only stops autograd; logging or publishing the values is
outside Opaque's DP accounting. General token-level metrics live here
(shared); preference reward telemetry (``reward_metrics``) is DPO-specific and
lives in
:mod:`opaque.api.alignment.dpo.metric`.
"""

from opaque.api.alignment.metric._token import entropy_from_logits, mean_token_accuracy

__all__ = ["entropy_from_logits", "mean_token_accuracy"]
