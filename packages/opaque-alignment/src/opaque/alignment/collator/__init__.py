"""opaque.alignment.collator façade — the preference (DPO) collator.

Unpaired-preference (KTO) → :mod:`opaque.alignment.kto.collator`; language-
modeling (SFT) → :mod:`opaque.alignment.sft.collator`.
"""

from opaque.api.alignment.collator import preference_collator

__all__ = ["preference_collator"]
