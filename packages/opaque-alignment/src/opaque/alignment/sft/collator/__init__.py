"""opaque.alignment.sft.collator façade — the language-modeling collator.

Output schema (:class:`LMBatch`) lives in
:mod:`opaque.alignment.sft.collator.types`.
"""

from opaque.alignment.sft.collator import types
from opaque.api.alignment.sft.collator import language_modeling_collator

__all__ = ["language_modeling_collator", "types"]
