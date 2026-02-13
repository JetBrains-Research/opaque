# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""vmap compatibility patches for HuggingFace Transformers models.

This module provides automatic model detection and patching for vmap compatibility.
Each model architecture has its own patcher that applies only the necessary patches.

Usage:
    from opaque.compat.transformers import patch_model, vmap_compat

    # Option 1: Explicit patching
    patch_model(model)
    # ... use clipped_grad ...

    # Option 2: Context manager
    with vmap_compat(model):
        grads, state = grad_fn(...)
"""

from opaque.compat.transformers.registry import (
    patch_model,
    unpatch_model,
    is_patched,
    vmap_compat,
    get_model_architecture,
    list_supported_architectures,
    SUPPORTED_ARCHITECTURES,
)
from opaque.compat.transformers._global_patches import (
    apply_global_patches,
    remove_global_patches,
    is_globally_patched,
)

__all__ = [
    "patch_model",
    "unpatch_model",
    "is_patched",
    "vmap_compat",
    "get_model_architecture",
    "list_supported_architectures",
    "SUPPORTED_ARCHITECTURES",
    "apply_global_patches",
    "remove_global_patches",
    "is_globally_patched",
]
