"""Torch registrations for the portable authoring contract."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optree
import torch
from torch.func import grad_and_value as _grad_and_value
from torch.func import vmap as _vmap

from opaque.api.engine import autodiff, ops
from opaque.api.engine.random._engine import RngKey, generator_from_key

if TYPE_CHECKING:
    from opaque.api.engine.primitive import Primitive


def is_array(value: Any) -> bool:
    return isinstance(value, torch.Tensor)


def dtype(value: Any) -> torch.dtype:
    return value.dtype


def shape(value: Any) -> tuple[int, ...]:
    return tuple(value.shape)


def is_floating(value: Any) -> bool:
    value_dtype = value if isinstance(value, torch.dtype) else value.dtype
    return torch.is_floating_point(torch.empty((), dtype=value_dtype))


def is_low_precision(value: Any) -> bool:
    value_dtype = value if isinstance(value, torch.dtype) else value.dtype
    return value_dtype in (torch.float16, torch.bfloat16)


def is_complex(value: Any) -> bool:
    value_dtype = value if isinstance(value, torch.dtype) else value.dtype
    return torch.is_complex(torch.empty((), dtype=value_dtype))


def float32() -> torch.dtype:
    return torch.float32


def real_dtype(value: Any) -> torch.dtype:
    value_dtype = value if isinstance(value, torch.dtype) else value.dtype
    return torch.empty((), dtype=value_dtype).real.dtype


def scalar(value: Any, *, dtype: Any = None, like: Any = None) -> torch.Tensor:
    device = like.device if like is not None else None
    return torch.tensor(value, dtype=dtype, device=device)


def zeros(shape: Any, *, dtype: Any = None, like: Any = None) -> torch.Tensor:
    device = like.device if like is not None else None
    return torch.zeros(shape, dtype=dtype, device=device)


def zeros_like(value: Any) -> torch.Tensor:
    return torch.zeros_like(value)


def ones_like(value: Any) -> torch.Tensor:
    return torch.ones_like(value)


def astype(value: Any, value_dtype: Any) -> torch.Tensor:
    return value.to(value_dtype)


def clone(value: Any) -> torch.Tensor:
    return value.clone()


def detach(value: Any) -> torch.Tensor:
    return value.detach()


def transfer(value: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
    return value.to(*args, **kwargs)


def scalar_item(value: Any) -> Any:
    return value.item()


def sqrt(value: Any) -> torch.Tensor:
    return torch.sqrt(value)


def square(value: Any) -> torch.Tensor:
    return torch.square(value)


def abs(value: Any) -> torch.Tensor:
    return torch.abs(value)


def add(left: Any, right: Any) -> torch.Tensor:
    return torch.add(left, right)


def subtract(left: Any, right: Any) -> torch.Tensor:
    return torch.subtract(left, right)


def multiply(left: Any, right: Any) -> torch.Tensor:
    return torch.multiply(left, right)


def divide(left: Any, right: Any) -> torch.Tensor:
    return torch.divide(left, right)


def sum(value: Any, axis: Any = None, dtype: Any = None) -> torch.Tensor:
    if axis is None:
        return torch.sum(value, dtype=dtype)
    return torch.sum(value, dim=axis, dtype=dtype)


def greater(left: Any, right: Any) -> torch.Tensor:
    return torch.gt(left, right)


def minimum(left: Any, right: Any) -> torch.Tensor:
    return torch.minimum(left, right)


def maximum(left: Any, right: Any) -> torch.Tensor:
    return torch.maximum(left, right)


def where(condition: Any, left: Any, right: Any) -> torch.Tensor:
    return torch.where(condition, left, right)


def isfinite(value: Any) -> torch.Tensor:
    return torch.isfinite(value)


def all(value: Any, axis: Any = None) -> torch.Tensor:
    return torch.all(value, dim=axis) if axis is not None else torch.all(value)


def nan_to_num(value: Any) -> torch.Tensor:
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def clamp(value: Any, lo: Any = None, hi: Any = None) -> torch.Tensor:
    return torch.clamp(value, min=lo, max=hi)


def concatenate(values: Any, axis: int = 0) -> torch.Tensor:
    return torch.cat(tuple(values), dim=axis)


def slice_array(value: Any, slices: Any) -> torch.Tensor:
    return value[slices]


def promote_dtype(first: Any, second: Any) -> torch.dtype:
    return torch.promote_types(first, second)


def grad_and_value(*args: Any, **kwargs: Any) -> Any:
    return _grad_and_value(*args, **kwargs)


def vmap(
    fn: Any, in_axes: Any = 0, out_axes: Any = 0, randomness: str = "error"
) -> Any:
    return _vmap(fn, in_dims=in_axes, out_dims=out_axes, randomness=randomness)


def tree_map(fn: Any, *trees: Any) -> Any:
    return optree.tree_map(fn, *trees)


def tree_flatten(tree: Any) -> tuple[list[Any], Any]:
    leaves, treedef = optree.tree_flatten(tree)
    return list(leaves), treedef


def tree_flatten_with_paths(tree: Any) -> tuple[list[Any], list[Any], Any]:
    from opaque.api.engine.pytree import param_path

    paths, leaves, treedef = optree.tree_flatten_with_path(tree)
    return [param_path(path) for path in paths], list(leaves), treedef


def tree_unflatten(treedef: Any, leaves: list[Any]) -> Any:
    return optree.tree_unflatten(treedef, leaves)


def tree_leaves(tree: Any) -> list[torch.Tensor]:
    leaves, _ = optree.tree_flatten(tree)
    return [leaf for leaf in leaves if is_array(leaf)]


def tree_structure(tree: Any) -> Any:
    return optree.tree_structure(tree)


def normal(
    rng_key: RngKey, shape: Any, *, dtype: Any = None, like: Any = None
) -> torch.Tensor:
    resolved_dtype = dtype or (like.dtype if like is not None else torch.float32)
    device = like.device if like is not None else None
    return torch.randn(
        shape,
        dtype=resolved_dtype,
        device=device,
        generator=generator_from_key(rng_key),
    )


def _register(primitive: Primitive, implementation: Any) -> None:
    if not primitive.supports("torch"):
        primitive.register("torch", implementation)


def register_core_primitives() -> None:
    """Idempotently register Torch implementations for every core primitive."""
    for primitive, implementation in (
        (ops.is_array, is_array),
        (ops.dtype, dtype),
        (ops.shape, shape),
        (ops.is_floating, is_floating),
        (ops.is_low_precision, is_low_precision),
        (ops.is_complex, is_complex),
        (ops.float32, float32),
        (ops.real_dtype, real_dtype),
        (ops.scalar, scalar),
        (ops.zeros, zeros),
        (ops.zeros_like, zeros_like),
        (ops.ones_like, ones_like),
        (ops.astype, astype),
        (ops.clone, clone),
        (ops.detach, detach),
        (ops.transfer, transfer),
        (ops.scalar_item, scalar_item),
        (ops.sqrt, sqrt),
        (ops.square, square),
        (ops.abs, abs),
        (ops.add, add),
        (ops.subtract, subtract),
        (ops.multiply, multiply),
        (ops.divide, divide),
        (ops.sum, sum),
        (ops.greater, greater),
        (ops.minimum, minimum),
        (ops.maximum, maximum),
        (ops.where, where),
        (ops.isfinite, isfinite),
        (ops.all, all),
        (ops.nan_to_num, nan_to_num),
        (ops.clamp, clamp),
        (ops.concatenate, concatenate),
        (ops.slice_array, slice_array),
        (ops.promote_dtype, promote_dtype),
        (autodiff.grad_and_value, grad_and_value),
        (autodiff.vmap, vmap),
    ):
        _register(primitive, implementation)

    from opaque.api.engine import pytree
    from opaque.api.engine.random import _engine as random_engine

    for primitive, implementation in (
        (pytree._tree_map, tree_map),
        (pytree._tree_flatten, tree_flatten),
        (pytree._tree_flatten_with_paths, tree_flatten_with_paths),
        (pytree._tree_unflatten, tree_unflatten),
        (pytree._tree_leaves, tree_leaves),
        (pytree._tree_structure, tree_structure),
        (random_engine._normal, normal),
    ):
        _register(primitive, implementation)


__all__ = ["register_core_primitives"]
