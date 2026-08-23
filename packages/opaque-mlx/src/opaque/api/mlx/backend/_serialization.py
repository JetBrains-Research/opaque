"""MLX exact-type serialization handlers."""

from __future__ import annotations

from typing import Any

import numpy as np

import mlx.core as mx
from opaque.api.base.serialization import register_serializer


def _check_shape(saved: tuple[int, ...], expected: tuple[int, ...]) -> None:
    if tuple(saved) != tuple(expected):
        raise ValueError(
            f"state_dict value has shape {tuple(saved)}; template expects "
            f"{tuple(expected)}. Restore is template-driven: rebuild the "
            "template from the configuration the checkpoint was written with."
        )


def _array_save(obj: mx.array) -> dict[str, Any]:
    mx.eval(obj)
    return {"": np.array(obj, copy=True)}


def _array_load(template: mx.array, state: dict[str, Any]) -> mx.array:
    saved = state.get("")
    if saved is None:
        return template
    if not isinstance(saved, (mx.array, np.ndarray)):
        raise TypeError(
            f"state_dict value expected an MLX array, got {type(saved).__name__}"
        )
    _check_shape(saved.shape, template.shape)
    return mx.array(saved, dtype=template.dtype)


def register_mlx_serialization() -> None:
    """Register MLX arrays with the base serialization registry."""
    register_serializer(mx.array, _array_save, _array_load)


__all__ = ["register_mlx_serialization"]
