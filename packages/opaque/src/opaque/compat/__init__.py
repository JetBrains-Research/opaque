# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Compatibility patches for external libraries.

Currently supports:
- transformers: HuggingFace Transformers models (vmap + kernel optimizations)

Environment variables (each accepts "all" or comma-separated names):
  OPAQUE_SKIP_COMPAT_PATCHES=all (or transformers)
  OPAQUE_SKIP_TRANSFORMERS_PATCHES=all (or vmap,kernels)
  OPAQUE_SKIP_TRANSFORMERS_VMAP_PATCHES=all (or shared,standard,gemma2,phi3)
  OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES=all (or swiglu,rope,ce,fused_ce,lora)
"""

import os

_is_patched = False


def apply_compat_patches() -> None:
    """Apply all compatibility patches for supported libraries.

    Currently patches:
    - HuggingFace Transformers (vmap compatibility + Triton kernel optimizations)

    Controlled by: OPAQUE_SKIP_COMPAT_PATCHES ("all" or "transformers")
    """
    global _is_patched

    if _is_patched:
        return

    raw_skip = os.environ.get("OPAQUE_SKIP_COMPAT_PATCHES", "")
    skip = {entry.strip().lower() for entry in raw_skip.split(",") if entry.strip()}

    if "all" in skip:
        _is_patched = True
        return

    if "transformers" not in skip:
        try:
            from opaque.compat.transformers import apply_transformers_patches

            apply_transformers_patches()
        except ImportError:
            pass

    _is_patched = True


def is_compat_patched() -> bool:
    """Check if compatibility patches have been applied."""
    return _is_patched


__all__ = ["apply_compat_patches", "is_compat_patched"]
