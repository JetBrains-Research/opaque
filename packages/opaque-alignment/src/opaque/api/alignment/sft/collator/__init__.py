"""SFT collator impl — the language-modeling collator factory.

``language_modeling_collator`` is a factory returning a stateless collate
callable (AGENTS.md rule 9). Its output schema (:class:`LMBatch`) lives in
:mod:`opaque.api.alignment.sft.collator.types`.
"""

from opaque.api.alignment.sft.collator._language_modeling import (
    language_modeling_collator,
)

__all__ = ["language_modeling_collator"]
