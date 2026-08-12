"""Torch registrations for the portable authoring contract."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optree

import torch
from opaque.api.engine import autodiff, ops, pytree
from opaque.api.engine.backend import KnownBackend
from opaque.api.engine.primitive import BackendProvider
from opaque.api.engine.random import _engine as random_engine
from opaque.api.torch.random import generator_from_key
from torch.func import grad_and_value as _grad_and_value
from torch.func import vmap as _vmap

if TYPE_CHECKING:
    from opaque.api.engine.random._engine import RngKey

_TORCH = BackendProvider(KnownBackend.TORCH)


@_TORCH.implements(ops.is_array)
def is_array(value: Any) -> bool:
    return isinstance(value, torch.Tensor)


@_TORCH.implements(ops.dtype)
def dtype(value: Any) -> torch.dtype:
    return value.dtype


@_TORCH.implements(ops.shape)
def shape(value: Any) -> tuple[int, ...]:
    return tuple(value.shape)


@_TORCH.implements(ops.is_floating)
def is_floating(value: Any) -> bool:
    value_dtype = value if isinstance(value, torch.dtype) else value.dtype
    return torch.is_floating_point(torch.empty((), dtype=value_dtype))


@_TORCH.implements(ops.is_low_precision)
def is_low_precision(value: Any) -> bool:
    value_dtype = value if isinstance(value, torch.dtype) else value.dtype
    return value_dtype in (torch.float16, torch.bfloat16)


@_TORCH.implements(ops.is_complex)
def is_complex(value: Any) -> bool:
    value_dtype = value if isinstance(value, torch.dtype) else value.dtype
    return torch.is_complex(torch.empty((), dtype=value_dtype))


@_TORCH.implements(ops.float32)
def float32() -> torch.dtype:
    return torch.float32


@_TORCH.implements(ops.real_dtype)
def real_dtype(value: Any) -> torch.dtype:
    value_dtype = value if isinstance(value, torch.dtype) else value.dtype
    return torch.empty((), dtype=value_dtype).real.dtype


@_TORCH.implements(ops.scalar)
def scalar(value: Any, *, dtype: Any = None, like: Any = None) -> torch.Tensor:
    device = like.device if like is not None else None
    return torch.tensor(value, dtype=dtype, device=device)


@_TORCH.implements(ops.zeros)
def zeros(shape: Any, *, dtype: Any = None, like: Any = None) -> torch.Tensor:
    device = like.device if like is not None else None
    return torch.zeros(shape, dtype=dtype, device=device)


@_TORCH.implements(ops.zeros_like)
def zeros_like(value: Any) -> torch.Tensor:
    return torch.zeros_like(value)


@_TORCH.implements(ops.ones_like)
def ones_like(value: Any) -> torch.Tensor:
    return torch.ones_like(value)


@_TORCH.implements(ops.astype)
def astype(value: Any, value_dtype: Any) -> torch.Tensor:
    return value.to(value_dtype)


@_TORCH.implements(ops.clone)
def clone(value: Any) -> torch.Tensor:
    return value.clone()


@_TORCH.implements(ops.detach)
def detach(value: Any) -> torch.Tensor:
    return value.detach()


@_TORCH.implements(ops.transfer)
def transfer(value: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
    return value.to(*args, **kwargs)


@_TORCH.implements(ops.scalar_item)
def scalar_item(value: Any) -> Any:
    return value.item()


@_TORCH.implements(ops.sqrt)
def sqrt(value: Any) -> torch.Tensor:
    return torch.sqrt(value)


@_TORCH.implements(ops.square)
def square(value: Any) -> torch.Tensor:
    return torch.square(value)


@_TORCH.implements(ops.abs)
def abs(value: Any) -> torch.Tensor:
    return torch.abs(value)


@_TORCH.implements(ops.add)
def add(left: Any, right: Any) -> torch.Tensor:
    return torch.add(left, right)


@_TORCH.implements(ops.subtract)
def subtract(left: Any, right: Any) -> torch.Tensor:
    return torch.subtract(left, right)


@_TORCH.implements(ops.multiply)
def multiply(left: Any, right: Any) -> torch.Tensor:
    return torch.multiply(left, right)


@_TORCH.implements(ops.divide)
def divide(left: Any, right: Any) -> torch.Tensor:
    return torch.divide(left, right)


@_TORCH.implements(ops.sum)
def sum(value: Any, axis: Any = None, dtype: Any = None) -> torch.Tensor:
    if axis is None:
        return torch.sum(value, dtype=dtype)
    return torch.sum(value, dim=axis, dtype=dtype)


@_TORCH.implements(ops.greater)
def greater(left: Any, right: Any) -> torch.Tensor:
    return torch.gt(left, right)


@_TORCH.implements(ops.minimum)
def minimum(left: Any, right: Any) -> torch.Tensor:
    return torch.minimum(left, right)


@_TORCH.implements(ops.maximum)
def maximum(left: Any, right: Any) -> torch.Tensor:
    return torch.maximum(left, right)


@_TORCH.implements(ops.where)
def where(condition: Any, left: Any, right: Any) -> torch.Tensor:
    return torch.where(condition, left, right)


@_TORCH.implements(ops.isfinite)
def isfinite(value: Any) -> torch.Tensor:
    return torch.isfinite(value)


@_TORCH.implements(ops.all)
def all(value: Any, axis: Any = None) -> torch.Tensor:
    return torch.all(value, dim=axis) if axis is not None else torch.all(value)


@_TORCH.implements(ops.nan_to_num)
def nan_to_num(value: Any) -> torch.Tensor:
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


@_TORCH.implements(ops.clamp)
def clamp(value: Any, lo: Any = None, hi: Any = None) -> torch.Tensor:
    return torch.clamp(value, min=lo, max=hi)


@_TORCH.implements(ops.concatenate)
def concatenate(values: Any, axis: int = 0) -> torch.Tensor:
    return torch.cat(tuple(values), dim=axis)


@_TORCH.implements(ops.slice_array)
def slice_array(value: Any, slices: Any) -> torch.Tensor:
    return value[slices]


@_TORCH.implements(ops.promote_dtype)
def promote_dtype(first: Any, second: Any) -> torch.dtype:
    return torch.promote_types(first, second)


@_TORCH.implements(autodiff._grad_and_value_transform)
def grad_and_value(*args: Any, **kwargs: Any) -> Any:
    return _grad_and_value(*args, **kwargs)


@_TORCH.implements(autodiff._vmap_transform)
def vmap(
    fn: Any, in_axes: Any = 0, out_axes: Any = 0, randomness: str = "error"
) -> Any:
    return _vmap(fn, in_dims=in_axes, out_dims=out_axes, randomness=randomness)


@_TORCH.implements(pytree.tree_map)
def tree_map(fn: Any, *trees: Any) -> Any:
    return optree.tree_map(fn, *trees)


@_TORCH.implements(pytree.tree_flatten)
def tree_flatten(tree: Any) -> tuple[list[Any], Any]:
    leaves, treedef = optree.tree_flatten(tree)
    return list(leaves), treedef


@_TORCH.implements(pytree.tree_flatten_with_paths)
def tree_flatten_with_paths(tree: Any) -> tuple[list[Any], list[Any], Any]:
    from opaque.api.engine.pytree import param_path

    paths, leaves, treedef = optree.tree_flatten_with_path(tree)
    return [param_path(path) for path in paths], list(leaves), treedef


@_TORCH.implements(pytree.tree_unflatten)
def tree_unflatten(treedef: Any, leaves: list[Any]) -> Any:
    return optree.tree_unflatten(treedef, leaves)


@_TORCH.implements(pytree.tree_leaves)
def tree_leaves(tree: Any) -> list[torch.Tensor]:
    leaves, _ = optree.tree_flatten(tree)
    return [leaf for leaf in leaves if is_array(leaf)]


@_TORCH.implements(pytree.tree_structure)
def tree_structure(tree: Any) -> Any:
    return optree.tree_structure(tree)


@_TORCH.implements(random_engine.normal)
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


__all__: list[str] = []
