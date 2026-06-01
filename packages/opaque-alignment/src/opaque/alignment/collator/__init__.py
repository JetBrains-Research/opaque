"""opaque.alignment.collator façade — re-exports preference collator factories.

The language-modeling (SFT) collator lives under
:mod:`opaque.alignment.sft.collator`.
"""

from opaque.api.alignment.collator import (
    preference_collator,
    unpaired_preference_collator,
)

__all__ = [
    "preference_collator",
    "unpaired_preference_collator",
]
