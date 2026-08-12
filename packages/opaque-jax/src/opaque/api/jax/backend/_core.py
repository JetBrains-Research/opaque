"""JAX implementation of Opaque's neutral compute backend protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jax import tree_util
from opaque.api.engine import autodiff, ops, pytree
from opaque.api.engine.backend import KnownBackend
from opaque.api.engine.primitive import BackendProvider
from opaque.api.engine.pytree import param_path
from opaque.api.engine.random import _engine as random_engine

if TYPE_CHECKING:
    from collections.abc import Callable

    from opaque.api.engine.random._engine import RngKey


_JAX = BackendProvider(KnownBackend.JAX)


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
    """Stable identity for the JAX provider."""

    name = KnownBackend.JAX.value


def _unsupported_randomness(randomness: str) -> None:
    if randomness != "error":
        raise ValueError(
            "JAX vmap supports only Opaque's randomness='error' mode; "
            f"got {randomness!r}."
        )


@_JAX.implements(autodiff._grad_and_value_transform)
def grad_and_value(
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


@_JAX.implements(autodiff._vmap_transform)
def vmap(
    fn: Any, in_axes: Any = 0, out_axes: Any = 0, randomness: str = "error"
) -> Any:
    _unsupported_randomness(randomness)
    return jax.vmap(fn, in_axes=in_axes, out_axes=out_axes)


@_JAX.implements(ops.is_array)
def is_array(value: Any) -> bool:
    return isinstance(value, (jax.Array, jax.core.Tracer))


@_JAX.implements(ops.dtype)
def dtype(value: Any) -> Any:
    return value.dtype


@_JAX.implements(ops.shape)
def shape(value: Any) -> tuple[int, ...]:
    return tuple(value.shape)


@_JAX.implements(ops.is_floating)
def is_floating(value: Any) -> bool:
    value_dtype = value.dtype if is_array(value) else value
    return bool(jnp.issubdtype(value_dtype, jnp.floating))


@_JAX.implements(ops.is_low_precision)
def is_low_precision(value: Any) -> bool:
    value_dtype = value.dtype if is_array(value) else value
    return jnp.dtype(value_dtype) in (
        jnp.dtype(jnp.float16),
        jnp.dtype(jnp.bfloat16),
    )


@_JAX.implements(ops.is_complex)
def is_complex(value: Any) -> bool:
    if is_array(value):
        return bool(jnp.iscomplexobj(value))
    return bool(jnp.issubdtype(value, jnp.complexfloating))


@_JAX.implements(ops.float32)
def float32() -> Any:
    return jnp.float32


@_JAX.implements(ops.real_dtype)
def real_dtype(value: Any) -> Any:
    value_dtype = value.dtype if is_array(value) else value
    return jnp.empty((), dtype=value_dtype).real.dtype


@_JAX.implements(ops.scalar)
def scalar(value: Any, *, dtype: Any = None, like: Any = None) -> Any:
    del like
    return jnp.asarray(value, dtype=dtype)


@_JAX.implements(ops.zeros)
def zeros(value_shape: Any, *, dtype: Any = None, like: Any = None) -> Any:
    del like
    return jnp.zeros(value_shape, dtype=dtype)


@_JAX.implements(ops.zeros_like)
def zeros_like(value: Any) -> Any:
    return jnp.zeros_like(value)


@_JAX.implements(ops.ones_like)
def ones_like(value: Any) -> Any:
    return jnp.ones_like(value)


@_JAX.implements(ops.astype)
def astype(value: Any, value_dtype: Any) -> Any:
    return value.astype(value_dtype)


@_JAX.implements(ops.clone)
def clone(value: Any) -> Any:
    return jnp.array(value, copy=True)


@_JAX.implements(ops.detach)
def detach(value: Any) -> Any:
    return jax.lax.stop_gradient(value)


@_JAX.implements(ops.transfer)
def transfer(value: Any, *args: Any, **kwargs: Any) -> Any:
    if "dtype" in kwargs:
        return value.astype(kwargs["dtype"])
    if args and hasattr(args[0], "dtype"):
        return value.astype(args[0])
    if args:
        return jax.device_put(value, args[0])
    return value


@_JAX.implements(ops.scalar_item)
def scalar_item(value: Any) -> Any:
    return value.item()


@_JAX.implements(ops.sqrt)
def sqrt(value: Any) -> Any:
    return jnp.sqrt(value)


@_JAX.implements(ops.square)
def square(value: Any) -> Any:
    return jnp.square(value)


@_JAX.implements(ops.abs)
def abs(value: Any) -> Any:
    return jnp.abs(value)


@_JAX.implements(ops.add)
def add(left: Any, right: Any) -> Any:
    return jnp.add(left, right)


@_JAX.implements(ops.subtract)
def subtract(left: Any, right: Any) -> Any:
    return jnp.subtract(left, right)


@_JAX.implements(ops.multiply)
def multiply(left: Any, right: Any) -> Any:
    return jnp.multiply(left, right)


@_JAX.implements(ops.divide)
def divide(left: Any, right: Any) -> Any:
    return jnp.divide(left, right)


@_JAX.implements(ops.sum)
def sum(value: Any, axis: Any = None, dtype: Any = None) -> Any:
    return jnp.sum(value, axis=axis, dtype=dtype)


@_JAX.implements(ops.greater)
def greater(left: Any, right: Any) -> Any:
    return jnp.greater(left, right)


@_JAX.implements(ops.minimum)
def minimum(left: Any, right: Any) -> Any:
    return jnp.minimum(left, right)


@_JAX.implements(ops.maximum)
def maximum(left: Any, right: Any) -> Any:
    return jnp.maximum(left, right)


@_JAX.implements(ops.where)
def where(condition: Any, left: Any, right: Any) -> Any:
    return jnp.where(condition, left, right)


@_JAX.implements(ops.isfinite)
def isfinite(value: Any) -> Any:
    return jnp.isfinite(value)


@_JAX.implements(ops.all)
def all(value: Any, axis: Any = None) -> Any:
    return jnp.all(value, axis=axis)


@_JAX.implements(ops.nan_to_num)
def nan_to_num(value: Any) -> Any:
    return jnp.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


@_JAX.implements(ops.clamp)
def clamp(value: Any, lo: Any = None, hi: Any = None) -> Any:
    if lo is not None:
        value = jnp.maximum(value, lo)
    if hi is not None:
        value = jnp.minimum(value, hi)
    return value


@_JAX.implements(ops.concatenate)
def concatenate(values: Any, axis: int = 0) -> Any:
    return jnp.concatenate(tuple(values), axis=axis)


@_JAX.implements(ops.slice_array)
def slice_array(value: Any, slices: Any) -> Any:
    return value[slices]


@_JAX.implements(ops.expand_dims)
def expand_dims(value: Any, axis: int) -> Any:
    return jnp.expand_dims(value, axis=axis)


@_JAX.implements(ops.squeeze)
def squeeze(value: Any, axis: int | None = None) -> Any:
    return jnp.squeeze(value, axis=axis)


@_JAX.implements(ops.promote_dtype)
def promote_dtype(left: Any, right: Any) -> Any:
    return jnp.promote_types(left, right)


@_JAX.implements(pytree.tree_map)
def tree_map(fn: Callable[..., Any], *trees: Any) -> Any:
    return tree_util.tree_map(fn, *trees)


@_JAX.implements(pytree.tree_flatten)
def tree_flatten(tree: Any) -> tuple[list[Any], Any]:
    leaves, treedef = tree_util.tree_flatten(tree)
    return list(leaves), treedef


@_JAX.implements(pytree.tree_flatten_with_paths)
def tree_flatten_with_paths(tree: Any) -> tuple[list[Any], list[Any], Any]:
    path_leaves, treedef = tree_util.tree_flatten_with_path(tree)
    paths = [
        param_path([_path_component(entry) for entry in path])
        for path, _ in path_leaves
    ]
    leaves = [leaf for _, leaf in path_leaves]
    return paths, leaves, treedef


@_JAX.implements(pytree.tree_unflatten)
def tree_unflatten(treedef: Any, leaves: list[Any]) -> Any:
    return tree_util.tree_unflatten(treedef, leaves)


@_JAX.implements(pytree.tree_leaves)
def tree_leaves(tree: Any) -> list[Any]:
    return [leaf for leaf in tree_util.tree_leaves(tree) if is_array(leaf)]


@_JAX.implements(pytree.tree_structure)
def tree_structure(tree: Any) -> Any:
    return tree_util.tree_structure(tree)


@_JAX.implements(random_engine.normal)
def normal(
    rng_key: RngKey,
    shape: Any,
    *,
    dtype: Any = None,
    like: Any = None,
) -> Any:
    resolved_dtype = dtype or (like.dtype if like is not None else jnp.float32)
    return jax.random.normal(
        jax.random.key(rng_key.seed), shape=shape, dtype=resolved_dtype
    )


__all__ = ["JaxBackend"]
