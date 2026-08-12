"""JAX implementation of Opaque's neutral compute backend protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jax import tree_util
from opaque.api.engine.pytree import param_path

if TYPE_CHECKING:
    from collections.abc import Callable

    from opaque.api.engine.random._engine import RngKey


def _path_component(entry: Any) -> str | int:
    """Extract an engine-compatible component from a JAX pytree key entry."""
    if isinstance(entry, tree_util.DictKey):
        component = entry.key
    elif isinstance(entry, tree_util.SequenceKey):
        component = entry.idx
    elif isinstance(entry, tree_util.GetAttrKey):
        component = entry.name
    else:
        component = getattr(entry, "key", entry)

    if not isinstance(component, (str, int)):
        raise TypeError(
            "JAX pytree path components must be str or int; "
            f"got {type(component).__name__}"
        )
    return component


class JaxBackend:
    """JAX-backed implementation of the five-primitive backend surface."""

    name = "jax"
    float32 = jnp.float32

    # --- autodiff ---
    def value_and_grad(
        self,
        fn: Callable[..., Any],
        argnums: int | tuple[int, ...] = 0,
        has_aux: bool = False,
    ) -> Callable[..., Any]:
        """Adapt JAX's ``(value, grad)`` convention to ``(grad, value)``."""
        transformed = jax.value_and_grad(fn, argnums=argnums, has_aux=has_aux)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            value, grad = transformed(*args, **kwargs)
            if has_aux:
                loss, aux = value
                return grad, (loss, aux)
            return grad, value

        return wrapped

    # --- vectorization ---
    def vmap(
        self,
        fn: Callable[..., Any],
        in_axes: Any = 0,
        out_axes: Any = 0,
        randomness: str = "error",
    ) -> Callable[..., Any]:
        """Vectorize with JAX; its API has no separate randomness policy."""
        del randomness
        return jax.vmap(fn, in_axes=in_axes, out_axes=out_axes)

    # --- pytree ---
    def tree_map(self, fn: Callable[..., Any], *trees: Any) -> Any:
        return tree_util.tree_map(fn, *trees)

    def tree_flatten(self, tree: Any) -> tuple[list[Any], Any]:
        leaves, treedef = tree_util.tree_flatten(tree)
        return list(leaves), treedef

    def tree_flatten_with_paths(self, tree: Any) -> tuple[list[Any], list[Any], Any]:
        path_leaves, treedef = tree_util.tree_flatten_with_path(tree)
        paths = [
            param_path([_path_component(entry) for entry in path])
            for path, _ in path_leaves
        ]
        leaves = [leaf for _, leaf in path_leaves]
        return paths, leaves, treedef

    def tree_unflatten(self, treedef: Any, leaves: list[Any]) -> Any:
        return tree_util.tree_unflatten(treedef, leaves)

    def tree_leaves(self, tree: Any) -> list[Any]:
        return [leaf for leaf in tree_util.tree_leaves(tree) if self.is_array(leaf)]

    # --- array math (elementwise + reduction + dtype helpers) ---
    def is_array(self, x: Any) -> bool:
        return isinstance(x, (jax.Array, jax.core.Tracer))

    def is_floating(self, x: Any) -> bool:
        dtype = x.dtype if self.is_array(x) else x
        return bool(jnp.issubdtype(dtype, jnp.floating))

    def is_low_precision(self, x: Any) -> bool:
        dtype = x.dtype if self.is_array(x) else x
        return jnp.dtype(dtype) in (jnp.dtype(jnp.float16), jnp.dtype(jnp.bfloat16))

    def is_complex(self, x: Any) -> bool:
        if self.is_array(x):
            return bool(jnp.iscomplexobj(x))
        return bool(jnp.issubdtype(x, jnp.complexfloating))

    def sqrt(self, x: Any) -> Any:
        return jnp.sqrt(x)

    def square(self, x: Any) -> Any:
        return jnp.square(x)

    def sum(self, x: Any, axis: Any = None, dtype: Any = None) -> Any:
        return jnp.sum(x, axis=axis, dtype=dtype)

    def minimum(self, a: Any, b: Any) -> Any:
        return jnp.minimum(a, b)

    def maximum(self, a: Any, b: Any) -> Any:
        return jnp.maximum(a, b)

    def where(self, cond: Any, a: Any, b: Any) -> Any:
        return jnp.where(cond, a, b)

    def isfinite(self, x: Any) -> Any:
        return jnp.isfinite(x)

    def nan_to_num(self, x: Any) -> Any:
        return jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def clamp(self, x: Any, lo: Any = None, hi: Any = None) -> Any:
        if lo is not None:
            x = jnp.maximum(x, lo)
        if hi is not None:
            x = jnp.minimum(x, hi)
        return x

    def zeros_like(self, x: Any) -> Any:
        return jnp.zeros_like(x)

    def concatenate(self, xs: Any, axis: int = 0) -> Any:
        return jnp.concatenate(tuple(xs), axis=axis)

    def astype(self, x: Any, dtype: Any) -> Any:
        return x.astype(dtype)

    def scalar(self, value: Any, *, dtype: Any = None, like: Any = None) -> Any:
        del like
        return jnp.asarray(value, dtype=dtype)

    def promote_dtype(self, a: Any, b: Any) -> Any:
        return jnp.promote_types(a, b)

    # --- rng ---
    def generator(self, key: RngKey) -> Any:
        return jax.random.key(key.seed)

    def normal(self, shape: Any, *, dtype: Any, generator: Any) -> Any:
        return jax.random.normal(generator, shape=shape, dtype=dtype)


__all__ = ["JaxBackend"]
