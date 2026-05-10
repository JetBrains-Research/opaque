"""Patches torch façade — re-exports from ``opaque.api.patches.torch``."""

from opaque.api.patches.torch import apply_checkpoint_patch, is_checkpoint_patched

__all__ = [
    "apply_checkpoint_patch",
    "is_checkpoint_patched",
]
