"""DPO metrics impl — preference reward telemetry.

``reward_metrics`` returns detached, un-noised values; release is outside
Opaque's DP accounting. Token metrics live in
:mod:`opaque.api.alignment.metric`.
"""

from opaque.api.alignment.dpo.metric._reward import reward_metrics

__all__ = ["reward_metrics"]
