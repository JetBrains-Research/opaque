"""MLX registrations for optional execution transforms."""

from __future__ import annotations

import warnings
from typing import Any

import mlx.core as mx
from opaque.api.engine import execution
from opaque.api.engine.backend import KnownBackend
from opaque.api.engine.primitive import BackendProvider

_MLX = BackendProvider(KnownBackend.MLX)


@_MLX.implements(execution._compile_transform)
def _compile_transform(fn: Any) -> Any:
    return mx.compile(fn)


@_MLX.implements(execution._checkpoint_transform)
def _checkpoint_transform(fn: Any) -> Any:
    return mx.checkpoint(fn)


@_MLX.implements(execution._optimize_saved_activations_transform)
def _optimize_saved_activations_transform(fn: Any) -> Any:
    _warn_unified_memory()

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    return wrapper


def _warn_unified_memory() -> None:
    """Emit a one-time explanation that MLX uses unified memory.

    Unified memory removes the separate host/device placement problem, so
    ``optimize_saved_activations`` is an identity transform on MLX. Total
    activation storage is not reduced.
    """
    if getattr(_warn_unified_memory, "_emitted", False):
        return
    warnings.warn(
        "MLX uses unified memory, so optimize_saved_activations does not move "
        "activations to separate host memory. The transform returns fn unchanged; "
        "total activation memory is not reduced.",
        stacklevel=3,
    )
    _warn_unified_memory._emitted = True  # type: ignore[attr-defined]


__all__ = [
    "_compile_transform",
    "_checkpoint_transform",
    "_optimize_saved_activations_transform",
]
