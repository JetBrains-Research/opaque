"""Collator factories impl — preference (DPO).

The unpaired-preference (KTO) collator lives under
:mod:`opaque.api.alignment.kto.collator`; the language-modeling (SFT) collator
under :mod:`opaque.api.alignment.sft.collator`.
"""

from opaque.api.alignment.collator._preference import preference_collator

__all__ = ["preference_collator"]
