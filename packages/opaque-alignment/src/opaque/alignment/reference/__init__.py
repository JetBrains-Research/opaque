"""opaque.alignment.reference façade — re-exports reference-model helpers.

``RefSpec`` lives in :mod:`opaque.alignment.reference.types`.
"""

from opaque.api.alignment.reference import (
    compute_ref_logprobs_for_dataset,
    ema_update_reference,
    null_ref_context,
    with_disabled_adapter,
)

__all__ = [
    "compute_ref_logprobs_for_dataset",
    "null_ref_context",
    "with_disabled_adapter",
    "ema_update_reference",
]
