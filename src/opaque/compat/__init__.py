# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Compatibility patches for various libraries.

Currently supports:
- transformers: HuggingFace Transformers models

Usage:
    from opaque.compat import patch_model, vmap_compat

    # Option 1: Explicit patching
    patch_model(model)
    grads, state = clipped_grad(...)
    unpatch_model(model)

    # Option 2: Context manager
    with vmap_compat(model):
        grads, state = clipped_grad(...)
"""

from opaque.compat.transformers import (
    patch_model,
    unpatch_model,
    is_patched,
    vmap_compat,
    get_model_architecture,
    list_supported_architectures,
    SUPPORTED_ARCHITECTURES,
)

# Legacy aliases for backward compatibility
patch_for_vmap = patch_model
unpatch = unpatch_model

__all__ = [
    "patch_model",
    "unpatch_model",
    "is_patched",
    "vmap_compat",
    "get_model_architecture",
    "list_supported_architectures",
    "SUPPORTED_ARCHITECTURES",
    # Legacy aliases
    "patch_for_vmap",
    "unpatch",
]
