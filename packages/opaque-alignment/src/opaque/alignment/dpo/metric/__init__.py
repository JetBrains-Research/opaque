"""opaque.alignment.dpo.metric façade — preference reward telemetry.

General token-level metrics live in the shared impl
:mod:`opaque.api.alignment.metric`.
"""

from opaque.api.alignment.dpo.metric import reward_metrics

__all__ = ["reward_metrics"]
