"""Opaque performance: fused kernels and PyTorch-version performance patches.

This package provides semantically-lossless speed/memory wins:

- :mod:`opaque.performance.kernels` — fused Triton kernels (with pure
  PyTorch fallbacks) for cross-entropy, RoPE, SwiGLU/GeGLU, and LoRA.
- :mod:`opaque.performance.torch.checkpoint` — gradient-checkpointing
  patches that make ``torch.utils.checkpoint`` compatible with
  ``vmap(grad(...))``.

Opt-in patching
---------------
The gradient-checkpointing patches mutate PyTorch internals; apply them
explicitly with :func:`patch_all` or :func:`patch_checkpoint`. The
kernels themselves are pure function calls and require no patching.

Environment::

    OPAQUE_SKIP_PYTORCH_PATCHES=all              # skip all torch patches
    OPAQUE_SKIP_PYTORCH_CHECKPOINT_PATCHES=all   # skip just the checkpoint patch
"""

from opaque.core._env import parse_skip_env
from opaque.performance.torch.checkpoint import (
    is_checkpoint_patched,
    patch_checkpoint,
    unpatch_checkpoint,
)

__version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    "patch_all",
    "patch_checkpoint",
    "unpatch_checkpoint",
    "is_checkpoint_patched",
]


def patch_all() -> None:
    """Apply all opt-in performance patches.

    Currently applies the gradient-checkpointing patches. Honors
    ``OPAQUE_SKIP_PYTORCH_PATCHES`` (values: ``all`` or ``checkpoint``).
    """
    skip = parse_skip_env("OPAQUE_SKIP_PYTORCH_PATCHES")
    if "all" in skip:
        return
    if "checkpoint" not in skip:
        patch_checkpoint()
