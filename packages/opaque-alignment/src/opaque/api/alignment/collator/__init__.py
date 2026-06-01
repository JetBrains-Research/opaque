"""Collator factories impl — preference, unpaired-preference.

Each public symbol is a factory function returning a collator callable
(AGENTS.md rule 9). The language-modeling (SFT) collator lives under
:mod:`opaque.api.alignment.sft.collator`.
"""

from opaque.api.alignment.collator._preference import preference_collator
from opaque.api.alignment.collator._unpaired_preference import (
    unpaired_preference_collator,
)

__all__ = [
    "preference_collator",
    "unpaired_preference_collator",
]
