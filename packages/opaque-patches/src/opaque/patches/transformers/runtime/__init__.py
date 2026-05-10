"""Patches transformers runtime façade."""

from opaque.api.patches.transformers.runtime import (
    apply_collator_patches,
    apply_masking_patches,
)

__all__ = [
    "apply_collator_patches",
    "apply_masking_patches",
]
