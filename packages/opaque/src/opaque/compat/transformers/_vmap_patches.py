# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Vmap compatibility patches for HuggingFace Transformers models.

Makes HF models work with torch.func.vmap(grad()) for DP-SGD per-example
gradients. Applied at `import opaque` time.

Patched groups:
- shared: causal mask (masking_utils), repeat_kv (sdpa_attention)
- standard: eager attention for LLaMA, Mistral, Qwen2, Qwen3, Granite, Cohere, Cohere2
- gemma2: softcap-aware attention
- phi3: DynamicCache compatibility, repeat_kv

Skip all with: OPAQUE_SKIP_TRANSFORMERS_VMAP_PATCHES=all
Skip specific groups: OPAQUE_SKIP_TRANSFORMERS_VMAP_PATCHES=gemma2,phi3
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_is_vmap_patched = False


def apply_vmap_patches() -> None:
    """Apply vmap compatibility patches to HuggingFace Transformers models.

    Patches at module/class level for:
    - shared: causal mask, repeat_kv utilities
    - standard: eager attention for standard models
    - gemma2: softcap attention
    - phi3: DynamicCache compatibility

    No-op when OPAQUE_SKIP_TRANSFORMERS_VMAP_PATCHES=all.
    """
    global _is_vmap_patched

    if _is_vmap_patched:
        return

    skip = os.environ.get("OPAQUE_SKIP_TRANSFORMERS_VMAP_PATCHES", "").lower().split(",")
    if "all" in skip:
        _is_vmap_patched = True
        return

    patched = []

    if "shared" not in skip:
        from opaque.compat.transformers._shared import apply_shared_patches

        apply_shared_patches()
        patched.append("shared")

    if "standard" not in skip:
        from opaque.compat.transformers._standard_models import (
            apply_standard_model_patches,
        )

        apply_standard_model_patches()
        patched.append("standard")

    if "gemma2" not in skip:
        from opaque.compat.transformers._gemma2 import apply_gemma2_patches

        apply_gemma2_patches()
        patched.append("gemma2")

    if "phi3" not in skip:
        from opaque.compat.transformers._phi3 import apply_phi3_patches

        apply_phi3_patches()
        patched.append("phi3")

    if patched:
        logger.debug(f"opaque: Applied vmap patches: {', '.join(patched)}")

    _is_vmap_patched = True


def is_vmap_patched() -> bool:
    """Check if vmap compatibility patches have been applied."""
    return _is_vmap_patched
