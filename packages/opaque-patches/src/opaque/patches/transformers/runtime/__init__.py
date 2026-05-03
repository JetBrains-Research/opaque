# SPDX-License-Identifier: Apache-2.0
"""Global runtime patches for HuggingFace Transformers."""

from .collator import apply_collator_patches
from .loss_mapping import apply_loss_mapping_patch
from .masking import apply_masking_patches

__all__ = [
    "apply_collator_patches",
    "apply_loss_mapping_patch",
    "apply_masking_patches",
]
