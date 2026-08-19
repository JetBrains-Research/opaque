"""Torch registrations for optional execution transforms."""

from __future__ import annotations

from typing import Any

import torch
from opaque.api.engine import execution
from opaque.api.torch.backend._checkpoint_compat import apply_checkpoint_patch
from opaque.api.torch.backend._core import _TORCH


@_TORCH.implements(execution._compile_transform)
def _compile_transform(fn: Any) -> Any:
    return torch.compile(fn)


@_TORCH.implements(execution._checkpoint_transform)
def _checkpoint_transform(fn: Any) -> Any:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        apply_checkpoint_patch(vmap_checkpointing=True)
        return torch.utils.checkpoint.checkpoint(
            fn, *args, use_reentrant=False, **kwargs
        )

    return wrapper


@_TORCH.implements(execution._optimize_saved_activations_transform)
def _optimize_saved_activations_transform(fn: Any) -> Any:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        apply_checkpoint_patch(vmap_checkpointing=True)
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            return fn(*args, **kwargs)

    return wrapper


__all__ = [
    "_compile_transform",
    "_checkpoint_transform",
    "_optimize_saved_activations_transform",
]
