"""opaque.alignment.collator façade — re-exports collator factories.

Output-schema TypedDicts live in :mod:`opaque.alignment.collator.types`.
"""

from opaque.api.alignment.collator import (
    language_modeling_collator,
    preference_collator,
    unpaired_preference_collator,
)

__all__ = [
    "language_modeling_collator",
    "preference_collator",
    "unpaired_preference_collator",
]
