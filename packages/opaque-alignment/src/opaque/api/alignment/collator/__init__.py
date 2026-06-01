"""Collator factories impl — language-modeling, preference, unpaired.

Each public symbol is a factory function returning a collator callable
(AGENTS.md rule 9). Output schemas live in
:mod:`opaque.api.alignment.collator.types`.
"""

from opaque.api.alignment.collator._language_modeling import language_modeling_collator
from opaque.api.alignment.collator._preference import preference_collator
from opaque.api.alignment.collator._unpaired_preference import (
    unpaired_preference_collator,
)

__all__ = [
    "language_modeling_collator",
    "preference_collator",
    "unpaired_preference_collator",
]
