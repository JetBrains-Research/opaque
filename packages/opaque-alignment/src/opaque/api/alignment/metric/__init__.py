"""Shared alignment metrics impl — exact token-level telemetry.

These metrics are detached but un-noised; release is outside Opaque's DP
accounting. Preference metrics live in
:mod:`opaque.api.alignment.dpo.metric`.
"""

from opaque.api.alignment.metric._token import entropy_from_logits, mean_token_accuracy

__all__ = ["entropy_from_logits", "mean_token_accuracy"]
