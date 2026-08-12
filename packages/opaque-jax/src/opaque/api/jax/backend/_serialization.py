"""JAX exact-type serialization handlers for the engine provider."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from opaque.api.base.serialization import register_serializer


def _array_save(obj: jax.Array) -> dict[str, Any]:
    return {"": jnp.array(obj, copy=True)}


def _array_load(template: jax.Array, state: dict[str, Any]) -> jax.Array:
    saved = state.get("")
    if saved is None:
        return template
    if not isinstance(saved, jax.Array):
        raise TypeError(
            f"state_dict value expected a JAX array, got {type(saved).__name__}"
        )
    restored = jax.device_put(saved, template.device)
    return restored.astype(template.dtype)


def register_jax_serialization() -> None:
    """Register the native JAX array handler with the base registry."""
    register_serializer(jax.Array, _array_save, _array_load)


__all__ = ["register_jax_serialization"]
