"""DPO metrics impl — preference reward telemetry.

``reward_metrics`` returns detached but un-noised values computed from the
private batch. Detaching only stops autograd; logging or publishing the values
is outside Opaque's DP accounting. General token-level metrics (entropy,
accuracy) live in the shared :mod:`opaque.api.alignment.metric`.
"""

from opaque.api.alignment.dpo.metric._reward import reward_metrics

__all__ = ["reward_metrics"]
