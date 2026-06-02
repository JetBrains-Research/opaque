"""Logprob helpers impl — selective log-softmax + per-sequence completion logp.

Shared, internal primitives: ``selective_log_softmax`` (used by the SFT losses),
``sequence_logp`` (the eager DPO/causal-LM per-sequence logp), and
``fused_sequence_logp`` (its memory-efficient drop-in over hidden states, fused
through the opaque-patches linear-CE kernel with an eager fallback). The fused
variant is per-example: drive it with ``vmap(grad)``.
"""

from opaque.api.alignment.logprob._gather import selective_log_softmax
from opaque.api.alignment.logprob._sequence import (
    fused_sequence_logp,
    sequence_logp,
)

__all__ = [
    "selective_log_softmax",
    "sequence_logp",
    "fused_sequence_logp",
]
