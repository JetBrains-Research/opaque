"""opaque-alignment implementation namespace: DPO method.

Method-first layout (mirrors ``opaque.api.dpsgd`` / ``opaque.api.dpftrl`` and
the sibling ``opaque.api.alignment.sft``): DPO owns its loss family
(``dpo/loss``), preference collator (``dpo/collator``), fused-linear preference
kernel (``dpo/kernel``), reference-model handling (``dpo/reference``), reward
telemetry (``dpo/metric``), and preference prompt extraction (``dpo/data``).
Shared primitives (logprob, chat-template data, general token metrics) are
reimported from the shared ``opaque.api.alignment.*`` concerns at their use
sites.
"""

__all__: list[str] = []
