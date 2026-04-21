"""HuggingFace kernel patches shipped by ``opaque-performance``.

Applies Opaque's fused Triton kernels (SwiGLU, GeGLU, RoPE, cross-entropy,
fused linear+CE, LoRA) on top of HuggingFace Transformers model classes. Pure
performance: the model already runs correctly without these patches; they
reduce memory and compute.

Patches are applied automatically when :mod:`opaque.performance` is imported
(which itself opts into patching). Standalone users can also call
:func:`patch_all` here explicitly.

Disable with:

- ``OPAQUE_SKIP_PYTORCH_PATCHES=all`` — skip the whole performance package.
- ``OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES=all`` — skip only these kernels.
- ``OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES=swiglu,rope,ce,fused_ce,lora`` — selective.

Requires ``transformers`` to be importable; otherwise :func:`patch_all` is a
soft no-op (log at debug level).
"""

from __future__ import annotations

import logging

from opaque.performance.huggingface.kernel_patches import (
    apply_kernel_patches,
    is_kernel_patched,
    patch_lora_model,
)

logger = logging.getLogger(__name__)


def patch_all() -> None:
    """Apply the transformers kernel patches; no-op if transformers is missing."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        logger.debug(
            "opaque.performance.huggingface.patch_all: transformers is not installed; "
            "skipping kernel patches."
        )
        return
    apply_kernel_patches()


__all__ = [
    "patch_all",
    "apply_kernel_patches",
    "is_kernel_patched",
    "patch_lora_model",
]
