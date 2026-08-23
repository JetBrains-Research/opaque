"""MLX registrations for optional execution transforms."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from opaque.api.engine import execution
from opaque.api.mlx.backend._core import _MLX


@_MLX.implements(execution._compile_transform)
def _compile_transform(fn: Any) -> Any:
    return mx.compile(fn)


@_MLX.implements(execution._checkpoint_transform)
def _checkpoint_transform(fn: Any) -> Any:
    return mx.checkpoint(fn)


@_MLX.implements(execution._optimize_saved_activations_transform)
def _optimize_saved_activations_transform(fn: Any) -> Any:
    return fn


__all__ = [
    "_checkpoint_transform",
    "_compile_transform",
    "_optimize_saved_activations_transform",
]
