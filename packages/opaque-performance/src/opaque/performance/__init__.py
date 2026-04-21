"""Opaque performance: fused kernels and PyTorch-version performance patches.

This package provides semantically-lossless speed/memory wins:

- :mod:`opaque.performance.kernels` — fused Triton kernels (with pure
  PyTorch fallbacks) for cross-entropy, RoPE, SwiGLU/GeGLU, and LoRA.
- :mod:`opaque.performance.torch.checkpoint` — gradient-checkpointing
  patches that make ``torch.utils.checkpoint`` compatible with
  ``vmap(grad(...))``.
- :mod:`opaque.performance.huggingface` — fused-kernel patches wiring
  the kernels above into Transformers model classes.
- :mod:`opaque.performance.profiling` — memory / step-time profiler.

On-import patching
------------------
Importing :mod:`opaque.performance` runs :func:`patch_all`, which applies the
gradient-checkpointing patches and the Transformers kernel patches (the
latter soft-imports ``transformers`` and no-ops if it is missing).

Environment::

    OPAQUE_SKIP_PYTORCH_PATCHES=all                # skip all torch patches
    OPAQUE_SKIP_PYTORCH_CHECKPOINT_PATCHES=all     # skip just the checkpoint patch
    OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES=all    # skip the HF kernel patches
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from opaque.core._env import parse_skip_env
from opaque.performance.torch.checkpoint import (
    is_checkpoint_patched,
    patch_checkpoint,
)

try:
    __version__ = _pkg_version("opaque-performance")
except PackageNotFoundError:
    __version__ = "0.0.0"


def patch_all() -> None:
    """Apply all performance patches (idempotent).

    Honors:

    - ``OPAQUE_SKIP_PYTORCH_PATCHES`` (``all`` or ``checkpoint``) for the
      gradient-checkpointing patch.
    - ``OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES`` for the HF kernel patches
      (handled inside :mod:`opaque.performance.huggingface`).

    The Transformers kernel patches are soft-imports: if ``transformers`` is
    not installed they are silently skipped.
    """
    skip = parse_skip_env("OPAQUE_SKIP_PYTORCH_PATCHES")
    if "all" in skip:
        return
    if "checkpoint" not in skip:
        patch_checkpoint()

    # HF kernel patches (best-effort — transformers may not be installed).
    from opaque.performance.huggingface import patch_all as _patch_hf_kernels

    _patch_hf_kernels()


__all__ = [
    "__version__",
    "patch_all",
    "is_checkpoint_patched",
]
# `patch_checkpoint` is importable for explicit re-application but not
# in __all__ — env vars + patch_all() are the recommended paths.


# Auto-apply performance patches on import. Disable via
# OPAQUE_SKIP_PYTORCH_PATCHES=all.
patch_all()
