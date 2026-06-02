"""Shared alignment metrics impl — general token-level telemetry.

These are private internal state: they return detached tensors intended for
logging/accumulation, not for release. General
token-level metrics live here (shared); preference reward telemetry
(``reward_metrics``) is DPO-specific and lives in
:mod:`opaque.api.alignment.dpo.metric`.
"""

from opaque.api.alignment.metric._token import entropy_from_logits, mean_token_accuracy

__all__ = ["entropy_from_logits", "mean_token_accuracy"]
