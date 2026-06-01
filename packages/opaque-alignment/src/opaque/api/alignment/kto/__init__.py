"""opaque-alignment implementation namespace: KTO method (loss, collator, data).

Method-first layout (mirrors ``opaque.dpsgd`` / ``opaque.dpftrl``): KTO owns its
loss math (``kto/loss`` — ``kto_loss`` is Tier 2, ``apo_zero_unpaired`` Tier 1),
its unpaired-preference collator (``kto/collator``), and the completion rotation
for the KL term (``kto/data``). Shared primitives (logprob, metric, reference,
prompt / chat-template) are reimported from ``opaque.api.alignment.*``.
"""

__all__: list[str] = []
