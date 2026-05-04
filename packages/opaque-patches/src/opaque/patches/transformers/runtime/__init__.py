# SPDX-License-Identifier: Apache-2.0
"""Global runtime patches for HuggingFace Transformers."""

from .collator import apply_collator_patches
from .masking import apply_masking_patches

__all__ = [
    "apply_collator_patches",
    "apply_masking_patches",
]
