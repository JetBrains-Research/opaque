"""opaque-alignment implementation namespace: DPO method.

Method-first layout (mirrors ``opaque.api.dpsgd`` / ``opaque.api.dpftrl`` and
the sibling ``opaque.api.alignment.sft``): DPO owns its loss family
(``dpo/loss``), preference collator (``dpo/collator``), reference-model handling
(``dpo/reference``), reward telemetry (``dpo/metric``), and preference prompt
extraction (``dpo/data``). The fused DPO path is ``fused_sequence_logp`` (a
memory-efficient drop-in for ``sequence_logp``) in the shared
``opaque.api.alignment.logprob`` concern, composed with the ``dpo/loss``
per-pair heads. Shared primitives (logprob, chat-template data, general token
metrics) are reimported from the shared ``opaque.api.alignment.*`` concerns at
their use sites.
"""

__all__: list[str] = []
