"""opaque.alignment.collator façade — the preference (DPO) collator.

The language-modeling (SFT) collator lives under
:mod:`opaque.alignment.sft.collator`.
"""

from opaque.api.alignment.collator import preference_collator

__all__ = ["preference_collator"]
