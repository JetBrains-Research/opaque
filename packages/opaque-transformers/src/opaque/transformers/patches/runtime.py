"""Patches transformers runtime façade."""

from opaque.api.transformers.patches.runtime import (
    apply_collator_patches,
    apply_grouped_mm_patches,
    apply_masking_patches,
)

__all__ = [
    "apply_collator_patches",
    "apply_grouped_mm_patches",
    "apply_masking_patches",
]
