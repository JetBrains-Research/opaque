"""Collator factories impl — preference (DPO).

The language-modeling (SFT) collator lives under
:mod:`opaque.api.alignment.sft.collator`.
"""

from opaque.api.alignment.dpo.collator._preference import preference_collator

__all__ = ["preference_collator"]
