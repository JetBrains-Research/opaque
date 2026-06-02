"""opaque.alignment.dpo.logp façade — per-sequence completion log-probabilities.

The logp primitives the DPO per-pair heads (:mod:`opaque.alignment.dpo.loss`)
consume — kept here as a sibling of ``loss`` because a logp is *not* a loss:

- ``sequence_logp`` — eager per-sequence completion logp (from logits).
- ``fused_sequence_logp`` — its memory-efficient drop-in (from hidden states +
  the ``lm_head`` weight), fused through the opaque-patches linear-CE kernel with
  an eager fallback.

Both are per-example and ``torch.func``-composable; a DPO ``per_example_loss`` is
``head(sequence_logp(...) − ref_logp, …)`` (see ``examples/train_dpo.py``). The
lower-level ``selective_log_softmax`` building block stays internal under
``opaque.api.alignment.logprob`` for power users.
"""

from opaque.api.alignment.logprob import fused_sequence_logp, sequence_logp

__all__ = ["sequence_logp", "fused_sequence_logp"]
