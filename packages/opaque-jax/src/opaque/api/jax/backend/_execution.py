"""JAX registrations for optional execution transforms."""

from __future__ import annotations

from typing import Any

import jax
from jax import checkpoint_policies
from opaque.api.engine import execution
from opaque.api.engine.backend import KnownBackend
from opaque.api.engine.primitive import BackendProvider

_JAX = BackendProvider(KnownBackend.JAX)


@_JAX.implements(execution._compile_transform)
def _compile_transform(fn: Any) -> Any:
    return jax.jit(fn)


@_JAX.implements(execution._checkpoint_transform)
def _checkpoint_transform(fn: Any) -> Any:
    return jax.checkpoint(fn)


@_JAX.implements(execution._optimize_saved_activations_transform)
def _optimize_saved_activations_transform(fn: Any) -> Any:
    return jax.checkpoint(
        fn,
        policy=checkpoint_policies.offload_dot_with_no_batch_dims(
            "device", "pinned_host"
        ),
    )


__all__ = [
    "_compile_transform",
    "_checkpoint_transform",
    "_optimize_saved_activations_transform",
]
