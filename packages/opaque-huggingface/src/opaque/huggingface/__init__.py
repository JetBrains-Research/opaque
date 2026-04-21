"""Opaque HuggingFace integration.

Compatibility patches for HuggingFace Transformers — fixes that let supported
architectures run under ``vmap(grad(...))``. Performance / kernel patches
(SwiGLU, GeGLU, RoPE, fused CE, LoRA) live separately in
:mod:`opaque.performance.huggingface`.

Patches are applied automatically on import; disable with the
``OPAQUE_SKIP_TRANSFORMERS_*`` environment variables documented in
:mod:`opaque.huggingface.patches`. You can also call :func:`patch_all` to
apply / re-apply explicitly.
"""

from opaque.huggingface.patches import (
    apply_transformers_patches as _apply,
    is_transformers_patched as _is_patched,
    is_vmap_patched,
)

__version__ = "0.0.0.dev0"


def patch_all() -> None:
    """Apply all HuggingFace Transformers compatibility patches (idempotent)."""
    _apply()


def is_patched() -> bool:
    """Check whether :func:`patch_all` has already been applied."""
    return _is_patched()


__all__ = [
    "__version__",
    "patch_all",
    "is_patched",
    "is_vmap_patched",
]


# Auto-apply compatibility patches on import. Disable via
# OPAQUE_SKIP_TRANSFORMERS_PATCHES=all.
patch_all()
