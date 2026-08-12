"""Reusable neutral-backend clipping harness for cross-framework tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from opaque.api.engine.backend import use_backend
from opaque.api.engine.clipping._pytree import auto_scale_pytree, clip_pytree
from opaque.pytree import global_norm

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from opaque.api.engine.backend._protocol import Backend


@dataclass(frozen=True)
class HostBridge:
    """Evaluate a framework array and expose it through NumPy."""

    to_numpy: Callable[[Any], np.ndarray]


@dataclass(frozen=True)
class ClippingRun:
    """Outputs from a per-example neutral clipping composition."""

    per_example_grads: Any
    clipped_grads: Any
    clipped_norms: Any
    summed_grads: Any
    values: Any
    value_aux: Any


def run_clipping(
    backend: Backend,
    loss_fn: Callable[..., Any],
    params: Any,
    batch_x: Any,
    batch_y: Any,
    *,
    kind: Literal["fixed", "auto"],
    bound: float,
    gamma: float = 0.05,
) -> ClippingRun:
    """Differentiate, clip each example, and sum using only ``Backend`` primitives.

    MLX's ``vmap`` requires all output leaves to be arrays.  The clipping
    auxiliary includes a ``None`` group-norm field in scalar mode, so the
    vmap callback deliberately returns only the transformed gradient tree.
    Per-example norms are computed separately through the same backend seam.
    """
    with use_backend(backend):
        grad_and_value = backend.value_and_grad(loss_fn, has_aux=True)
        per_example_grads, (values, value_aux) = backend.vmap(
            grad_and_value, in_axes=(None, 0, 0)
        )(params, batch_x, batch_y)

        if kind == "fixed":

            def transform(grad: Any) -> Any:
                return clip_pytree(grad, bound)[0]

        else:

            def transform(grad: Any) -> Any:
                return auto_scale_pytree(grad, R=bound, gamma=gamma)[0]

        clipped_grads = backend.vmap(transform)(per_example_grads)
        clipped_norms = backend.vmap(global_norm)(clipped_grads)
        summed_grads = backend.tree_map(
            lambda leaf: backend.sum(leaf, axis=0), clipped_grads
        )

    return ClippingRun(
        per_example_grads=per_example_grads,
        clipped_grads=clipped_grads,
        clipped_norms=clipped_norms,
        summed_grads=summed_grads,
        values=values,
        value_aux=value_aux,
    )


__all__ = ["ClippingRun", "HostBridge", "run_clipping"]
