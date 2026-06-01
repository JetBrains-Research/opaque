"""Logprob helpers impl — selective log-softmax + per-sequence completion logp.

Shared, internal primitives (no public façade): consumed by sibling impl
modules — ``selective_log_softmax`` by the SFT losses
(:mod:`opaque.api.alignment.sft.loss`), ``sequence_logp`` by the DPO kernel
(:mod:`opaque.api.alignment.dpo.kernel`). ``sequence_logp`` is re-exported to
users through the DPO façade (:mod:`opaque.alignment.dpo`), its public consumer.
All functions are pure and vmap-safe (see plan §7.5).
"""

from opaque.api.alignment.logprob._gather import selective_log_softmax
from opaque.api.alignment.logprob._sequence import sequence_logp

__all__ = ["selective_log_softmax", "sequence_logp"]
