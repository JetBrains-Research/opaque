"""DPO metrics impl — preference reward telemetry.

``reward_metrics`` returns detached tensors for logging/accumulation, not for
release. General token-level metrics (entropy,
accuracy) live in the shared :mod:`opaque.api.alignment.metric`.
"""

from opaque.api.alignment.dpo.metric._reward import reward_metrics

__all__ = ["reward_metrics"]
