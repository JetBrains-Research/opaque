"""PyTorch-version behavioral performance patches.

Currently exposes:

- :mod:`opaque.performance.torch.checkpoint` — gradient-checkpointing
  patches that make ``torch.utils.checkpoint`` compatible with
  ``vmap(grad(...))``.

These are temporary shims until the corresponding fixes land in upstream
PyTorch.
"""

from opaque.performance.torch.checkpoint import (
    is_checkpoint_patched,
    patch_checkpoint,
    unpatch_checkpoint,
)

__all__ = [
    "patch_checkpoint",
    "unpatch_checkpoint",
    "is_checkpoint_patched",
]
