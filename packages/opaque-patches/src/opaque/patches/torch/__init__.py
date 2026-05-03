"""PyTorch-version behavioral performance patches.

Currently exposes:

- :mod:`opaque.patches.torch.runtime` — gradient-checkpointing
  patches that make ``torch.utils.checkpoint`` compatible with
  ``vmap(grad(...))``.

These are temporary shims until the corresponding fixes land in upstream
PyTorch.
"""

from opaque.patches.torch.runtime import (
    apply_checkpoint_patch,
    is_checkpoint_patched,
)

__all__ = [
    "apply_checkpoint_patch",
    "is_checkpoint_patched",
]
