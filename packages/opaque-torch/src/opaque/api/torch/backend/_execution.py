"""Torch registrations for optional execution transforms."""

from __future__ import annotations

from typing import Any

import torch
from opaque.api.engine import execution
from opaque.api.torch.backend._core import _TORCH

# Neither transform installs the vmap-safe checkpoint patches. The engine binds
# an execution transform lazily, so this module's implementation first runs on
# the transform's *first invocation* — by then a surrounding ``torch.func``
# interpreter is already on the stack and has made its saved-tensor-hook check,
# so a patch applied here cannot take effect. Composing either transform under
# ``grad_and_value`` / ``vmap`` / ``clipped_grad`` therefore requires an earlier,
# explicit ``opaque.torch.checkpoint.apply_checkpoint_patch()``. Plain eager use
# needs no patch at all, so applying one from here would also mean patching
# torch globals for callers who never asked.


@_TORCH.implements(execution._compile_transform)
def _compile_transform(fn: Any) -> Any:
    return torch.compile(fn)


@_TORCH.implements(execution._checkpoint_transform)
def _checkpoint_transform(fn: Any) -> Any:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return torch.utils.checkpoint.checkpoint(
            fn, *args, use_reentrant=False, **kwargs
        )

    return wrapper


@_TORCH.implements(execution._optimize_saved_activations_transform)
def _optimize_saved_activations_transform(fn: Any) -> Any:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            return fn(*args, **kwargs)

    return wrapper


__all__ = [
    "_compile_transform",
    "_checkpoint_transform",
    "_optimize_saved_activations_transform",
]
