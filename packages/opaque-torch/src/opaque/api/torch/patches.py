# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""The Torch provider's global runtime patches.

These rebind PyTorch internals that ``torch.func`` cannot compose with as
shipped. They are applied explicitly, never on import: see
:func:`apply_runtime_patches`.

Higher layers own their own runtime patches and call this one first.
:func:`opaque.transformers.apply_runtime_patches` does exactly that — fixing
Hugging Face requires fixing torch, so it forwards its flags here before
applying the Hugging Face layer, and a caller needs only the one call.
"""

from __future__ import annotations

from typing import Any

from opaque.api.torch.backend._checkpoint_compat import apply_checkpoint_patch

__all__ = ["apply_runtime_patches"]


def apply_runtime_patches(*, compat: bool = True, **kwargs: Any) -> None:
    """Apply the Torch-core runtime patches.

    ``compat`` is the umbrella flag; each concern can be overridden by name.
    Concerns:

    - ``vmap_checkpointing`` — make non-reentrant gradient checkpointing
      compose under ``vmap(grad(...))``. See
      :func:`opaque.torch.checkpoint.apply_checkpoint_patch`, which is this
      concern's single-purpose entry point.

    Unknown keyword arguments are ignored, so a higher layer can forward its
    whole keyword set without filtering it first. Every patch is idempotent;
    calling this more than once is safe.
    """
    if kwargs.get("vmap_checkpointing", compat):
        apply_checkpoint_patch(vmap_checkpointing=True)
