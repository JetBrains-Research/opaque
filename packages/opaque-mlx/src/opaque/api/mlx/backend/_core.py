"""MLX implementation of Opaque's neutral compute backend protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optree

import mlx.core as mx
import mlx.utils as mx_utils
from opaque.api.engine import autodiff, ops, pytree
from opaque.api.engine.backend import KnownBackend
from opaque.api.engine.primitive import BackendProvider
from opaque.api.engine.pytree import param_path
from opaque.api.engine.random import _engine as random_engine

if TYPE_CHECKING:
    from collections.abc import Callable

    from opaque.api.engine.random._engine import RngKey


_MLX = BackendProvider(KnownBackend.MLX)


class MlxBackend:
    """Stable identity for the MLX provider."""

    name = KnownBackend.MLX.value


@_MLX.implements(autodiff._grad_and_value_transform)
def grad_and_value(
    fn: Callable[..., Any],
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
) -> Callable[..., Any]:
    """Adapt MLX's ``(value, grad)`` convention to ``(grad, value)``."""
    transformed = mx.value_and_grad(fn, argnums=argnums)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        value, grad = transformed(*args, **kwargs)
        if has_aux:
            loss, aux = value
            return grad, (loss, aux)
        return grad, value

    return wrapped


@_MLX.implements(autodiff._vmap_transform)
def vmap(
    fn: Callable[..., Any],
    in_axes: Any = 0,
    out_axes: Any = 0,
) -> Callable[..., Any]:
    return mx.vmap(fn, in_axes=in_axes, out_axes=out_axes)


@_MLX.implements(ops.is_array)
def is_array(value: Any) -> bool:
    return isinstance(value, mx.array)


@_MLX.implements(ops.dtype)
def dtype(value: Any) -> Any:
    return value.dtype


@_MLX.implements(ops.shape)
def shape(value: Any) -> tuple[int, ...]:
    return tuple(value.shape)


@_MLX.implements(ops.is_floating)
def is_floating(value: Any) -> bool:
    value_dtype = value if isinstance(value, mx.Dtype) else value.dtype
    return mx.issubdtype(value_dtype, mx.floating)


@_MLX.implements(ops.is_low_precision)
def is_low_precision(value: Any) -> bool:
    value_dtype = value if isinstance(value, mx.Dtype) else value.dtype
    return value_dtype in (mx.float16, mx.bfloat16)


@_MLX.implements(ops.is_complex)
def is_complex(value: Any) -> bool:
    value_dtype = value if isinstance(value, mx.Dtype) else value.dtype
    complex128 = getattr(mx, "complex128", mx.complex64)
    return value_dtype == mx.complex64 or value_dtype == complex128


@_MLX.implements(ops.float32)
def float32() -> Any:
    return mx.float32


@_MLX.implements(ops.real_dtype)
def real_dtype(value: Any) -> Any:
    value_dtype = value.dtype if is_array(value) else value
    return mx.real(mx.zeros((), dtype=value_dtype)).dtype


@_MLX.implements(ops.scalar)
def scalar(value: Any, *, dtype: Any = None, like: Any = None) -> Any:
    del like
    return mx.array(value, dtype=dtype)


@_MLX.implements(ops.zeros)
def zeros(value_shape: Any, *, dtype: Any = None, like: Any = None) -> Any:
    del like
    return mx.zeros(value_shape, dtype=dtype)


@_MLX.implements(ops.zeros_like)
def zeros_like(value: Any) -> Any:
    return mx.zeros_like(value)


@_MLX.implements(ops.ones_like)
def ones_like(value: Any) -> Any:
    return mx.ones_like(value)


@_MLX.implements(ops.astype)
def astype(value: Any, value_dtype: Any) -> Any:
    return value.astype(value_dtype)


@_MLX.implements(ops.clone)
def clone(value: Any) -> Any:
    return mx.array(value)


@_MLX.implements(ops.detach)
def detach(value: Any) -> Any:
    return value


@_MLX.implements(ops.transfer)
def transfer(value: Any, *args: Any, **kwargs: Any) -> Any:
    value_dtype = kwargs.get("dtype")
    if value_dtype is not None:
        return value.astype(value_dtype)
    if args and isinstance(args[0], mx.Dtype):
        return value.astype(args[0])
    return value


@_MLX.implements(ops.scalar_item)
def scalar_item(value: Any) -> Any:
    return value.item()


@_MLX.implements(ops.sqrt)
def sqrt(value: Any) -> Any:
    return mx.sqrt(value)


@_MLX.implements(ops.exp)
def exp(value: Any) -> Any:
    return mx.exp(value)


@_MLX.implements(ops.erf)
def erf(value: Any) -> Any:
    return mx.erf(value)


@_MLX.implements(ops.erfinv)
def erfinv(value: Any) -> Any:
    return mx.erfinv(value)


@_MLX.implements(ops.finfo_eps)
def finfo_eps(value_dtype: Any) -> float:
    return float(mx.finfo(value_dtype).eps)


@_MLX.implements(ops.rsqrt)
def rsqrt(value: Any) -> Any:
    return mx.rsqrt(value)


@_MLX.implements(ops.square)
def square(value: Any) -> Any:
    return mx.square(value)


@_MLX.implements(ops.abs)
def abs(value: Any) -> Any:
    return mx.abs(value)


@_MLX.implements(ops.add)
def add(left: Any, right: Any) -> Any:
    return mx.add(left, right)


@_MLX.implements(ops.subtract)
def subtract(left: Any, right: Any) -> Any:
    return mx.subtract(left, right)


@_MLX.implements(ops.multiply)
def multiply(left: Any, right: Any) -> Any:
    return mx.multiply(left, right)


@_MLX.implements(ops.divide)
def divide(left: Any, right: Any) -> Any:
    return mx.divide(left, right)


@_MLX.implements(ops.sum)
def sum(value: Any, axis: Any = None, dtype: Any = None) -> Any:
    if dtype is not None:
        value = value.astype(dtype)
    return mx.sum(value, axis=axis)


@_MLX.implements(ops.pow)
def pow(value: Any, exponent: Any) -> Any:
    return mx.power(value, exponent)


@_MLX.implements(ops.mean)
def mean(value: Any, axis: Any = None) -> Any:
    return mx.mean(value, axis=axis)


@_MLX.implements(ops.reciprocal)
def reciprocal(value: Any) -> Any:
    return mx.reciprocal(value)


@_MLX.implements(ops.accumulator_dtype)
def accumulator_dtype(value: Any, *, kind: str = "sum") -> Any:
    del kind
    return mx.float32


@_MLX.implements(ops.greater)
def greater(left: Any, right: Any) -> Any:
    return mx.greater(left, right)


@_MLX.implements(ops.minimum)
def minimum(left: Any, right: Any) -> Any:
    return mx.minimum(left, right)


@_MLX.implements(ops.maximum)
def maximum(left: Any, right: Any) -> Any:
    return mx.maximum(left, right)


@_MLX.implements(ops.where)
def where(condition: Any, left: Any, right: Any) -> Any:
    return mx.where(condition, left, right)


@_MLX.implements(ops.isfinite)
def isfinite(value: Any) -> Any:
    return mx.isfinite(value)


@_MLX.implements(ops.all)
def all(value: Any, axis: Any = None) -> Any:
    return mx.all(value, axis=axis)


@_MLX.implements(ops.nan_to_num)
def nan_to_num(value: Any) -> Any:
    return mx.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


@_MLX.implements(ops.clamp)
def clamp(value: Any, lo: Any = None, hi: Any = None) -> Any:
    if lo is not None:
        value = mx.maximum(value, lo)
    if hi is not None:
        value = mx.minimum(value, hi)
    return value


@_MLX.implements(ops.concatenate)
def concatenate(values: Any, axis: int = 0) -> Any:
    return mx.concatenate(tuple(values), axis=axis)


@_MLX.implements(ops.slice_array)
def slice_array(value: Any, slices: Any) -> Any:
    return value[slices]


@_MLX.implements(ops.expand_dims)
def expand_dims(value: Any, axis: int) -> Any:
    return mx.expand_dims(value, axis=axis)


@_MLX.implements(ops.squeeze)
def squeeze(value: Any, axis: int | None = None) -> Any:
    return mx.squeeze(value, axis=axis)


@_MLX.implements(ops.promote_dtype)
def promote_dtype(left: Any, right: Any) -> Any:
    return mx.result_type(left, right)


@_MLX.implements(pytree.tree_map)
def tree_map(fn: Callable[..., Any], *trees: Any) -> Any:
    return mx_utils.tree_map(fn, *trees)


@_MLX.implements(pytree.tree_flatten)
def tree_flatten(tree: Any) -> tuple[list[Any], Any]:
    # MLX's dotted flattening loses the distinction between a flat dotted key
    # and nested dictionaries, while optree preserves the structural treedef.
    leaves, treedef = optree.tree_flatten(tree)
    return list(leaves), treedef


@_MLX.implements(pytree.tree_flatten_with_paths)
def tree_flatten_with_paths(tree: Any) -> tuple[list[Any], list[Any], Any]:
    paths, leaves, treedef = optree.tree_flatten_with_path(tree)
    return [param_path(path) for path in paths], list(leaves), treedef


@_MLX.implements(pytree.tree_unflatten)
def tree_unflatten(treedef: Any, leaves: list[Any]) -> Any:
    return optree.tree_unflatten(treedef, leaves)


@_MLX.implements(pytree.tree_leaves)
def tree_leaves(tree: Any) -> list[Any]:
    return [leaf for _, leaf in mx_utils.tree_flatten(tree) if is_array(leaf)]


@_MLX.implements(pytree.tree_structure)
def tree_structure(tree: Any) -> Any:
    return optree.tree_structure(tree)


@_MLX.implements(random_engine.normal)
def normal(
    rng_key: RngKey,
    shape: Any,
    *,
    dtype: Any = None,
    like: Any = None,
) -> Any:
    resolved_dtype = dtype or (like.dtype if like is not None else mx.float32)
    return mx.random.normal(
        shape, dtype=resolved_dtype, key=mx.random.key(rng_key.seed)
    )


__all__ = ["MlxBackend"]
