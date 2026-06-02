"""opaque.alignment.dpo.reference façade — re-exports reference-model helpers."""

from opaque.api.alignment.dpo.reference import (
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
