# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Global patching orchestration for HuggingFace Transformers models.

This module coordinates applying all vmap-compatible patches at import time.
Patches are organized into three independent layers:

1. Shared utilities (masking_utils, sdpa_attention) - required by all models
2. Standard models (LLaMA, Mistral, Qwen2, etc.) - independent from each other
3. Custom models (Gemma2, etc.) - independent from each other but depend on shared

To add support for a new model with custom requirements:
1. Create _model_name.py with vmap implementations
2. Add apply_model_name_patches() function (with dependency note if needed)
3. Call it from apply_global_patches() below
"""

from opaque.compat.transformers._gemma2 import apply_gemma2_patches
from opaque.compat.transformers._kernel_patches import apply_kernel_patches
from opaque.compat.transformers._phi3 import apply_phi3_patches
from opaque.compat.transformers._shared import apply_shared_patches
from opaque.compat.transformers._standard_models import apply_standard_model_patches

# Track if patches have been applied
_is_patched = False


def apply_global_patches() -> None:
    """Apply all vmap compatibility and kernel optimization patches at import time.

    Orchestrates patching in four layers:
    1. Shared utilities - required by all models
    2. Standard models - can work independently after shared patches
    3. Custom models - can work independently after shared patches
    4. Triton kernel optimizations - replace RMSNorm/MLP with vmap-compatible Triton kernels

    Each model's patches are independent from other models.
    """
    global _is_patched

    if _is_patched:
        return

    # Layer 1: Patch shared utilities (required by all)
    apply_shared_patches()

    # Layer 2: Patch standard models (independent from each other)
    apply_standard_model_patches()

    # Layer 3: Patch custom models (independent from each other)
    apply_gemma2_patches()
    apply_phi3_patches()
    # Future custom models: add apply_*_patches() calls here

    # Layer 4: Triton kernel optimizations (when CUDA + Triton available)
    apply_kernel_patches()

    _is_patched = True


def is_globally_patched() -> bool:
    """Check if global patches are applied."""
    return _is_patched
