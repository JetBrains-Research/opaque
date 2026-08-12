"""MLX exact-type serialization handlers for the engine provider."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from opaque.api.base.serialization import register_serializer


def _array_save(obj: mx.array) -> dict[str, Any]:
    return {"": mx.array(obj)}


def _array_load(template: mx.array, state: dict[str, Any]) -> mx.array:
    saved = state.get("")
    if saved is None:
        return template
    if not isinstance(saved, mx.array):
        raise TypeError(
            f"state_dict value expected an MLX array, got {type(saved).__name__}"
        )
    return mx.array(saved, dtype=template.dtype)


def register_mlx_serialization() -> None:
    """Register the native MLX array handler with the base registry."""
    register_serializer(mx.array, _array_save, _array_load)


__all__ = ["register_mlx_serialization"]
