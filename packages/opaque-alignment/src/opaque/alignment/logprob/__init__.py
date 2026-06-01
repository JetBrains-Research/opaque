"""opaque.alignment.logprob façade — re-exports logprob helpers."""

from opaque.api.alignment.logprob import (
    get_batch_logps,
    selective_log_softmax,
    sequence_logp,
)

__all__ = ["selective_log_softmax", "sequence_logp", "get_batch_logps"]
