"""MLX implementations of the portable core primitives."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import optree

import mlx.core as mx
from opaque.api.engine import autodiff, ops, pytree
from opaque.api.engine.backend import KnownBackend
from opaque.api.engine.primitive import BackendProvider
from opaque.api.engine.random import _engine as random_engine

if TYPE_CHECKING:
    from opaque.api.engine.random._engine import RngKey

_MLX = BackendProvider(KnownBackend.MLX)
_PYTREE_NAMESPACE = "opaque.mlx"
_REDUCTION_BLOCK = 2048
_BLOCKED_REDUCTION_MIN = 4096


class MlxBackend:
    """Stable identity for the MLX provider."""

    name = KnownBackend.MLX.value


def _as_dtype(value: Any) -> mx.Dtype:
    return value.dtype if isinstance(value, mx.array) else value


def _placement(dtype: Any, like: Any) -> Any:
    return dtype if dtype is not None else (like.dtype if like is not None else None)


@_MLX.implements(ops.is_array)
def is_array(value: Any) -> bool:
    return isinstance(value, mx.array)


@_MLX.implements(ops.dtype)
def dtype(value: Any) -> mx.Dtype:
    return value.dtype


@_MLX.implements(ops.shape)
def shape(value: Any) -> tuple[int, ...]:
    return tuple(value.shape)


@_MLX.implements(ops.is_floating)
def is_floating(value: Any) -> bool:
    return mx.issubdtype(_as_dtype(value), mx.floating)


@_MLX.implements(ops.is_low_precision)
def is_low_precision(value: Any) -> bool:
    return _as_dtype(value) in (mx.float16, mx.bfloat16)


@_MLX.implements(ops.is_complex)
def is_complex(value: Any) -> bool:
    return mx.issubdtype(_as_dtype(value), mx.complexfloating)


@_MLX.implements(ops.float32)
def float32() -> mx.Dtype:
    return mx.float32


@_MLX.implements(ops.boolean)
def boolean() -> mx.Dtype:
    return mx.bool_


@_MLX.implements(ops.real_dtype)
def real_dtype(value: Any) -> mx.Dtype:
    value_dtype = _as_dtype(value)
    if value_dtype == mx.complex64:
        return mx.float32
    return value_dtype


@_MLX.implements(ops.scalar)
def scalar(value: Any, *, dtype: Any = None, like: Any = None) -> mx.array:
    return mx.array(value, dtype=_placement(dtype, like))


@_MLX.implements(ops.zeros)
def zeros(shape: Any, *, dtype: Any = None, like: Any = None) -> mx.array:
    return mx.zeros(shape, dtype=_placement(dtype, like))


@_MLX.implements(ops.zeros_like)
def zeros_like(value: Any) -> mx.array:
    return mx.zeros_like(value)


@_MLX.implements(ops.ones_like)
def ones_like(value: Any) -> mx.array:
    return mx.ones_like(value)


@_MLX.implements(ops.astype)
def astype(value: Any, value_dtype: Any) -> mx.array:
    return mx.astype(value, value_dtype)


@_MLX.implements(ops.clone)
def clone(value: Any) -> mx.array:
    return mx.array(value)


@_MLX.implements(ops.detach)
def detach(value: Any) -> mx.array:
    return mx.stop_gradient(value)


@_MLX.implements(ops.transfer)
def transfer(value: Any, *args: Any, **kwargs: Any) -> mx.array:
    value_dtype = kwargs.pop("dtype", None)
    if kwargs:
        raise TypeError(f"Unsupported MLX transfer options: {tuple(kwargs)}")
    if args and args != ("cpu",):
        raise TypeError("MLX transfer accepts only the unified-memory 'cpu' device")
    return mx.astype(value, value_dtype) if value_dtype is not None else mx.array(value)


@_MLX.implements(ops.scalar_item)
def scalar_item(value: Any) -> Any:
    return value.item()


@_MLX.implements(ops.sqrt)
def sqrt(value: Any) -> mx.array:
    return mx.sqrt(value)


@_MLX.implements(ops.exp)
def exp(value: Any) -> mx.array:
    return mx.exp(value)


@_MLX.implements(ops.erf)
def erf(value: Any) -> mx.array:
    return mx.erf(value)


@_MLX.implements(ops.erfinv)
def erfinv(value: Any) -> mx.array:
    return mx.erfinv(value)


@_MLX.implements(ops.finfo_eps)
def finfo_eps(value_dtype: Any) -> float:
    return float(mx.finfo(_as_dtype(value_dtype)).eps)


@_MLX.implements(ops.finfo_smallest_normal)
def finfo_smallest_normal(value_dtype: Any) -> float:
    return float(mx.finfo(_as_dtype(value_dtype)).smallest_normal)


@_MLX.implements(ops.to_host)
def to_host(value: Any) -> np.ndarray:
    mx.eval(value)
    return np.array(value, copy=True)


@_MLX.implements(ops.rsqrt)
def rsqrt(value: Any) -> mx.array:
    return mx.rsqrt(value)


@_MLX.implements(ops.square)
def square(value: Any) -> mx.array:
    return mx.square(value)


@_MLX.implements(ops.abs)
def abs(value: Any) -> mx.array:
    return mx.abs(value)


@_MLX.implements(ops.add)
def add(left: Any, right: Any) -> mx.array:
    return mx.add(left, right)


@_MLX.implements(ops.subtract)
def subtract(left: Any, right: Any) -> mx.array:
    return mx.subtract(left, right)


@_MLX.implements(ops.multiply)
def multiply(left: Any, right: Any) -> mx.array:
    return mx.multiply(left, right)


@_MLX.implements(ops.divide)
def divide(left: Any, right: Any) -> mx.array:
    return mx.divide(left, right)


@_MLX.implements(ops.sum)
def sum(value: Any, axis: Any = None, dtype: Any = None) -> mx.array:
    if dtype is not None:
        value = mx.astype(value, dtype)
    return mx.sum(value, axis=axis)


@_MLX.implements(ops.pow)
def pow(value: Any, exponent: Any) -> mx.array:
    return mx.power(value, exponent)


@_MLX.implements(ops.mean)
def mean(value: Any, axis: Any = None) -> mx.array:
    return mx.mean(value, axis=axis)


@_MLX.implements(ops.reciprocal)
def reciprocal(value: Any) -> mx.array:
    return mx.reciprocal(value)


@_MLX.implements(ops.accumulator_dtype)
def accumulator_dtype(value: Any, *, kind: str = "sum") -> mx.Dtype:
    del value, kind
    return mx.float32


@_MLX.implements(ops.amin)
def amin(value: Any, axis: Any = None) -> mx.array:
    return mx.min(value, axis=axis)


@_MLX.implements(ops.amax)
def amax(value: Any, axis: Any = None) -> mx.array:
    return mx.max(value, axis=axis)


@_MLX.implements(ops.greater)
def greater(left: Any, right: Any) -> mx.array:
    return mx.greater(left, right)


@_MLX.implements(ops.minimum)
def minimum(left: Any, right: Any) -> mx.array:
    return mx.minimum(left, right)


@_MLX.implements(ops.maximum)
def maximum(left: Any, right: Any) -> mx.array:
    return mx.maximum(left, right)


@_MLX.implements(ops.where)
def where(condition: Any, left: Any, right: Any) -> mx.array:
    return mx.where(condition, left, right)


@_MLX.implements(ops.isfinite)
def isfinite(value: Any) -> mx.array:
    return mx.isfinite(value)


@_MLX.implements(ops.all)
def all(value: Any, axis: Any = None) -> mx.array:
    return mx.all(value, axis=axis)


@_MLX.implements(ops.nan_to_num)
def nan_to_num(
    value: Any,
    *,
    nan: float = 0.0,
    posinf: float = 0.0,
    neginf: float = 0.0,
) -> mx.array:
    return mx.where(
        mx.isnan(value),
        nan,
        mx.where(
            mx.isposinf(value), posinf, mx.where(mx.isneginf(value), neginf, value)
        ),
    )


@_MLX.implements(ops.clamp)
def clamp(value: Any, lo: Any = None, hi: Any = None) -> mx.array:
    return mx.clip(value, lo, hi)


@_MLX.implements(ops.concatenate)
def concatenate(values: Any, axis: int = 0) -> mx.array:
    return mx.concatenate(tuple(values), axis=axis)


@_MLX.implements(ops.stack)
def stack(values: Any, axis: int = 0) -> mx.array:
    return mx.stack(tuple(values), axis=axis)


@_MLX.implements(ops.slice_array)
def slice_array(value: Any, slices: Any) -> mx.array:
    return value[slices]


@_MLX.implements(ops.expand_dims)
def expand_dims(value: Any, axis: int) -> mx.array:
    return mx.expand_dims(value, axis=axis)


@_MLX.implements(ops.squeeze)
def squeeze(value: Any, axis: int | None = None) -> mx.array:
    if axis is not None and value.shape[axis] != 1:
        return value
    return mx.squeeze(value, axis=axis)


@_MLX.implements(ops.promote_dtype)
def promote_dtype(first: Any, second: Any) -> mx.Dtype:
    return mx.result_type(_as_dtype(first), _as_dtype(second))


@_MLX.implements(autodiff._grad_and_value_transform)
def grad_and_value(
    fn: Any,
    argnums: Any = 0,
    has_aux: bool = False,
    values_only: bool = False,
) -> Any:
    del values_only
    if has_aux:

        def value_fn(*args: Any, **kwargs: Any) -> Any:
            value, _ = fn(*args, **kwargs)
            return value

        transformed = mx.value_and_grad(value_fn, argnums=argnums)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            value, gradient = transformed(*args, **kwargs)
            _, auxiliary = fn(*args, **kwargs)
            return gradient, (value, auxiliary)

        return wrapped

    transformed = mx.value_and_grad(fn, argnums=argnums)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        value, gradient = transformed(*args, **kwargs)
        return gradient, value

    return wrapped


@_MLX.implements(autodiff._vmap_transform)
def vmap(fn: Any, in_axes: Any = 0, out_axes: Any = 0, randomness: str = "same") -> Any:
    if randomness not in {"same", "different", "error"}:
        raise ValueError(f"Unsupported vmap randomness mode: {randomness!r}")
    return mx.vmap(fn, in_axes=in_axes, out_axes=out_axes)


@_MLX.implements(pytree.tree_map)
def tree_map(fn: Any, *trees: Any) -> Any:
    return optree.tree_map(fn, *trees, namespace=_PYTREE_NAMESPACE)


@_MLX.implements(pytree.tree_flatten)
def tree_flatten(tree: Any) -> tuple[list[Any], Any]:
    leaves, treedef = optree.tree_flatten(tree, namespace=_PYTREE_NAMESPACE)
    return list(leaves), treedef


@_MLX.implements(pytree.tree_flatten_with_paths)
def tree_flatten_with_paths(tree: Any) -> tuple[list[Any], list[Any], Any]:
    from opaque.api.engine.pytree import param_path

    paths, leaves, treedef = optree.tree_flatten_with_path(
        tree, namespace=_PYTREE_NAMESPACE
    )
    return [param_path(path) for path in paths], list(leaves), treedef


@_MLX.implements(pytree.tree_unflatten)
def tree_unflatten(treedef: Any, leaves: list[Any]) -> Any:
    return optree.tree_unflatten(treedef, leaves)


@_MLX.implements(pytree.tree_leaves)
def tree_leaves(tree: Any) -> list[mx.array]:
    leaves, _ = optree.tree_flatten(tree, namespace=_PYTREE_NAMESPACE)
    return [leaf for leaf in leaves if is_array(leaf)]


@_MLX.implements(pytree.tree_structure)
def tree_structure(tree: Any) -> Any:
    return optree.tree_structure(tree, namespace=_PYTREE_NAMESPACE)


def _blocked_reduction_terms(size: int) -> int:
    return _REDUCTION_BLOCK + (size + _REDUCTION_BLOCK - 1) // _REDUCTION_BLOCK


def _reduction_terms(leaves: list[mx.array]) -> int:
    widest = max((math.prod(leaf.shape) for leaf in leaves), default=0)
    if widest <= 1:
        return 0
    return (
        _blocked_reduction_terms(widest) if widest > _BLOCKED_REDUCTION_MIN else widest
    )


def _leaf_sq_sum(leaf: mx.array, compute_dtype: mx.Dtype) -> mx.array:
    if is_complex(leaf):
        real = mx.astype(leaf.real, compute_dtype)
        imaginary = mx.astype(leaf.imag, compute_dtype)
        squared = real * real + imaginary * imaginary
    else:
        cast = mx.astype(leaf, compute_dtype)
        squared = cast * cast
    flat = mx.reshape(mx.astype(squared, mx.float32), (-1,))
    size = flat.shape[0]
    if size <= _BLOCKED_REDUCTION_MIN:
        return mx.sum(flat)
    main = size // _REDUCTION_BLOCK * _REDUCTION_BLOCK
    total = mx.sum(mx.sum(mx.reshape(flat[:main], (-1, _REDUCTION_BLOCK)), axis=-1))
    return total + mx.sum(flat[main:]) if main < size else total


@_MLX.implements(pytree._squared_l2_norms)
def squared_l2_norms(
    leaves: list[mx.array], groups: list[str] | None, *, dtype: mx.Dtype
) -> tuple[mx.array, dict[str, mx.array]]:
    total: mx.array | None = None
    grouped: dict[str, mx.array] = {}
    for index, leaf in enumerate(leaves):
        squared = _leaf_sq_sum(leaf, dtype)
        total = squared if total is None else total + squared
        if groups is not None:
            group = groups[index]
            grouped[group] = (
                grouped.get(group, mx.zeros((), dtype=mx.float32)) + squared
            )
    return total if total is not None else mx.zeros((), dtype=mx.float32), grouped


@_MLX.implements(pytree._squared_l2_norm_roundoff)
def squared_l2_norm_roundoff(leaves: list[mx.array], *, dtype: mx.Dtype) -> float:
    compute_roundoff = mx.finfo(dtype).eps / 2.0 if is_floating(dtype) else 0.0
    accumulation_roundoff = mx.finfo(mx.float32).eps / 2.0
    return (
        compute_roundoff
        + (max(len(leaves), 1) + _reduction_terms(leaves)) * accumulation_roundoff
    ) / 2.0


@_MLX.implements(random_engine.normal)
def normal(
    rng_key: RngKey, shape: Any, *, dtype: Any = None, like: Any = None
) -> mx.array:
    resolved_dtype = _placement(dtype, like) or mx.float32
    return mx.random.normal(
        shape, dtype=resolved_dtype, key=mx.random.key(rng_key.seed)
    )


__all__ = ["MlxBackend"]
