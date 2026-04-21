# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""vmap compatibility and kernel optimization patches for HuggingFace Transformers.

Patching is opt-in: call :func:`opaque.huggingface.patch_all` (or the
umbrella :func:`opaque.patch_all`). Nothing is monkey-patched at import
time.

Environment variables:
    OPAQUE_SKIP_TRANSFORMERS_PATCHES         — "all" or "vmap,kernels,kv_cache,batchify,data"
    OPAQUE_SKIP_TRANSFORMERS_VMAP_PATCHES    — "all" or "shared,standard,gemma2,phi3"
    OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES  — "all" or "swiglu,rope,ce,fused_ce,lora"
    OPAQUE_SKIP_TRANSFORMERS_DATA_PATCHES    — "all" or "collator"

Supported models: GPT-2, LLaMA / Llama 3 (and LLaMA-based: DeepSeek,
Mistral, ...), Qwen2/3, Phi-3, Gemma / Gemma 2, Granite, Cohere /
Cohere 2.
"""

from opaque.core._env import parse_skip_env
from opaque.huggingface.patches._data_patches import apply_data_patches
from opaque.huggingface.patches._kernel_patches import (
    apply_kernel_patches,
    is_kernel_patched,
    patch_lora_model,
)
from opaque.huggingface.patches._shared import (
    apply_batchify_patches,
    apply_kv_cache_patches,
)
from opaque.huggingface.patches._vmap_patches import (
    apply_vmap_patches,
    is_vmap_patched,
)

_is_transformers_patched = False


def apply_transformers_patches() -> None:
    """Apply all HuggingFace Transformers patches (idempotent)."""
    global _is_transformers_patched

    if _is_transformers_patched:
        return

    skip = parse_skip_env("OPAQUE_SKIP_TRANSFORMERS_PATCHES")
    if "all" in skip:
        _is_transformers_patched = True
        return

    if "vmap" not in skip:
        apply_vmap_patches()

    if "kernels" not in skip:
        apply_kernel_patches()

    if "kv_cache" not in skip:
        apply_kv_cache_patches()

    if "vmap" not in skip and "batchify" not in skip:
        apply_batchify_patches()

    if "data" not in skip:
        apply_data_patches()

    _is_transformers_patched = True


def is_transformers_patched() -> bool:
    """Check if Transformers patches have been applied."""
    return _is_transformers_patched


__all__ = [
    "apply_transformers_patches",
    "apply_data_patches",
    "apply_kernel_patches",
    "apply_vmap_patches",
    "apply_batchify_patches",
    "apply_kv_cache_patches",
    "is_transformers_patched",
    "is_kernel_patched",
    "is_vmap_patched",
    "patch_lora_model",
]
