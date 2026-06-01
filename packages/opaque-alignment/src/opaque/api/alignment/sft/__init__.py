"""opaque-alignment implementation namespace: SFT method.

Method-first layout (mirrors ``opaque.dpsgd`` / ``opaque.dpftrl``): the SFT
method owns its loss math under ``sft/loss`` (and, as it lands, its collator
under ``sft/collator``). Shared primitives (logprob, metric, data) are
reimported from ``opaque.api.alignment.*`` at their use sites.
"""

__all__: list[str] = []
