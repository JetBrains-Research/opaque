# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""vmap / compatibility patches for HuggingFace Transformers.

Makes supported Transformers architectures work under ``vmap(grad(...))`` —
fixes to attention forwards, KV-cache interaction, and the Poisson-collator
compat patch. Purely correctness-oriented: no fused Triton kernels live here,
those ship in :mod:`opaque.performance.huggingface`.

Patches are applied automatically when :mod:`opaque.huggingface` is imported;
control with the ``OPAQUE_SKIP_TRANSFORMERS_*`` environment variables:

- ``OPAQUE_SKIP_TRANSFORMERS_PATCHES`` — ``all`` or any of
  ``vmap,kv_cache,batchify,data``.
- ``OPAQUE_SKIP_TRANSFORMERS_VMAP_PATCHES`` — ``all`` or any of
  ``shared,standard,gemma2,phi3``.
- ``OPAQUE_SKIP_TRANSFORMERS_DATA_PATCHES`` — ``all`` or ``collator``.

Supported models: GPT-2, LLaMA / Llama 3 (and LLaMA-based: DeepSeek,
Mistral, ...), Qwen2/3, Phi-3, Gemma / Gemma 2, Granite, Cohere /
Cohere 2.
"""

from opaque.core._env import parse_skip_env
from opaque.huggingface.patches._data_patches import apply_data_patches
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
    """Apply all compatibility patches (idempotent)."""
    global _is_transformers_patched

    if _is_transformers_patched:
        return

    skip = parse_skip_env("OPAQUE_SKIP_TRANSFORMERS_PATCHES")
    if "all" in skip:
        _is_transformers_patched = True
        return

    if "vmap" not in skip:
        apply_vmap_patches()

    if "kv_cache" not in skip:
        apply_kv_cache_patches()

    if "vmap" not in skip and "batchify" not in skip:
        apply_batchify_patches()

    if "data" not in skip:
        apply_data_patches()

    _is_transformers_patched = True


def is_transformers_patched() -> bool:
    """Check if compatibility patches have been applied."""
    return _is_transformers_patched


__all__ = [
    "apply_transformers_patches",
    "apply_data_patches",
    "apply_vmap_patches",
    "apply_batchify_patches",
    "apply_kv_cache_patches",
    "is_transformers_patched",
    "is_vmap_patched",
]
