# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Compatibility patches for PyTorch internals.

Currently supports:
- checkpoint: Gradient checkpointing under vmap(grad(...))

Environment variables:
  OPAQUE_SKIP_PYTORCH_PATCHES=all (or checkpoint)
  OPAQUE_SKIP_PYTORCH_CHECKPOINT_PATCHES=all
"""

from opaque.core._env import parse_skip_env
from opaque.compat.pytorch._checkpoint_patches import (
    apply_checkpoint_patches,
    is_checkpoint_patched,
)

_is_pytorch_patched = False


def apply_pytorch_patches() -> None:
    """Apply all PyTorch compatibility patches.

    Controlled by: OPAQUE_SKIP_PYTORCH_PATCHES ("all" or "checkpoint")
    """
    global _is_pytorch_patched

    if _is_pytorch_patched:
        return

    skip = parse_skip_env("OPAQUE_SKIP_PYTORCH_PATCHES")
    if "all" in skip:
        _is_pytorch_patched = True
        return

    if "checkpoint" not in skip:
        apply_checkpoint_patches()

    _is_pytorch_patched = True


def is_pytorch_patched() -> bool:
    """Check if PyTorch patches have been applied."""
    return _is_pytorch_patched


__all__ = [
    "apply_pytorch_patches",
    "apply_checkpoint_patches",
    "is_pytorch_patched",
    "is_checkpoint_patched",
]
