"""Logprob helpers impl — selective log-softmax + sequence/batch logp.

All functions are pure and vmap-safe (see plan §7.5).
"""

from opaque.api.alignment.logprob._batch import get_batch_logps
from opaque.api.alignment.logprob._gather import selective_log_softmax
from opaque.api.alignment.logprob._sequence import sequence_logp

__all__ = ["selective_log_softmax", "sequence_logp", "get_batch_logps"]
