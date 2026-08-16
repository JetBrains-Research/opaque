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
_PYTREE_NAMESPACE = "opaque.torch"


class TorchBackend:
    """Stable identity for the Torch provider."""

    name = KnownBackend.TORCH.value


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


@_TORCH.implements(ops.exp)
def exp(value: Any) -> torch.Tensor:
    return torch.exp(value)


@_TORCH.implements(ops.erf)
def erf(value: Any) -> torch.Tensor:
    return torch.erf(value)


@_TORCH.implements(ops.erfinv)
def erfinv(value: Any) -> torch.Tensor:
    return torch.erfinv(value)


@_TORCH.implements(ops.finfo_eps)
def finfo_eps(value_dtype: Any) -> float:
    return float(torch.finfo(value_dtype).eps)


@_TORCH.implements(ops.rsqrt)
def rsqrt(value: Any) -> torch.Tensor:
    return torch.rsqrt(value)


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


@_TORCH.implements(ops.pow)
def pow(value: Any, exponent: Any) -> torch.Tensor:
    return torch.pow(value, exponent)


@_TORCH.implements(ops.mean)
def mean(value: Any, axis: Any = None) -> torch.Tensor:
    if axis is None:
        return torch.mean(value)
    return torch.mean(value, dim=axis)


@_TORCH.implements(ops.reciprocal)
def reciprocal(value: Any) -> torch.Tensor:
    return torch.reciprocal(value)


@_TORCH.implements(ops.accumulator_dtype)
def accumulator_dtype(value: Any, *, kind: str = "sum") -> torch.dtype:
    del kind
    if ops.is_low_precision(value):
        return torch.float32
    if ops.is_array(value) and value.device.type == "mps":
        return torch.float32
    return torch.float64


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


@_TORCH.implements(ops.expand_dims)
def expand_dims(value: Any, axis: int) -> torch.Tensor:
    return torch.unsqueeze(value, dim=axis)


@_TORCH.implements(ops.squeeze)
def squeeze(value: Any, axis: int | None = None) -> torch.Tensor:
    return torch.squeeze(value) if axis is None else torch.squeeze(value, dim=axis)


@_TORCH.implements(ops.promote_dtype)
def promote_dtype(first: Any, second: Any) -> torch.dtype:
    return torch.promote_types(first, second)


@_TORCH.implements(autodiff._grad_and_value_transform)
def grad_and_value(*args: Any, **kwargs: Any) -> Any:
    return _grad_and_value(*args, **kwargs)


@_TORCH.implements(autodiff._vmap_transform)
def vmap(fn: Any, in_axes: Any = 0, out_axes: Any = 0) -> Any:
    return _vmap(fn, in_dims=in_axes, out_dims=out_axes)


@_TORCH.implements(pytree.tree_map)
def tree_map(fn: Any, *trees: Any) -> Any:
    return optree.tree_map(fn, *trees, namespace=_PYTREE_NAMESPACE)


@_TORCH.implements(pytree.tree_flatten)
def tree_flatten(tree: Any) -> tuple[list[Any], Any]:
    leaves, treedef = optree.tree_flatten(tree, namespace=_PYTREE_NAMESPACE)
    return list(leaves), treedef


@_TORCH.implements(pytree.tree_flatten_with_paths)
def tree_flatten_with_paths(tree: Any) -> tuple[list[Any], list[Any], Any]:
    from opaque.api.engine.pytree import param_path

    paths, leaves, treedef = optree.tree_flatten_with_path(
        tree, namespace=_PYTREE_NAMESPACE
    )
    return [param_path(path) for path in paths], list(leaves), treedef


@_TORCH.implements(pytree.tree_unflatten)
def tree_unflatten(treedef: Any, leaves: list[Any]) -> Any:
    return optree.tree_unflatten(treedef, leaves)


@_TORCH.implements(pytree.tree_leaves)
def tree_leaves(tree: Any) -> list[torch.Tensor]:
    leaves, _ = optree.tree_flatten(tree, namespace=_PYTREE_NAMESPACE)
    return [leaf for leaf in leaves if is_array(leaf)]


@_TORCH.implements(pytree.tree_structure)
def tree_structure(tree: Any) -> Any:
    return optree.tree_structure(tree, namespace=_PYTREE_NAMESPACE)


@_TORCH.implements(pytree._squared_l2_norms)
def squared_l2_norms(
    leaves: list[torch.Tensor],
    groups: list[str] | None,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    terms = [
        torch.sum(torch.square(torch.abs(leaf).to(dtype)), dtype=dtype)
        if torch.is_complex(leaf)
        else torch.sum(torch.square(leaf.to(dtype)), dtype=dtype)
        for leaf in leaves
    ]
    total = torch.stack(terms).sum(dtype=dtype)
    grouped: dict[str, torch.Tensor] = {}
    if groups is not None:
        grouped_terms: dict[str, list[torch.Tensor]] = {}
        for group, term in zip(groups, terms, strict=True):
            grouped_terms.setdefault(group, []).append(term)
        grouped = {
            group: torch.stack(values).sum(dtype=dtype)
            for group, values in grouped_terms.items()
        }
    return total, grouped


@_TORCH.implements(random_engine.normal)
def normal(
    rng_key: RngKey, shape: Any, *, dtype: Any = None, like: Any = None
) -> torch.Tensor:
    resolved_dtype = dtype or (like.dtype if like is not None else torch.float32)
    device = like.device if like is not None else None
    sample = torch.randn(
        shape,
        dtype=resolved_dtype,
        generator=generator_from_key(rng_key),
    )
    return sample.to(device=device) if device is not None else sample


# ---------------------------------------------------------------------------
# Optional array profiles
# ---------------------------------------------------------------------------


@_TORCH.implements(ops.asarray)
def asarray(value: Any, *, dtype: Any = None, like: Any = None) -> torch.Tensor:
    device = like.device if like is not None else None
    return torch.as_tensor(value, dtype=dtype, device=device)


@_TORCH.implements(ops.arange)
def arange(
    start: Any,
    stop: Any = None,
    step: Any = 1,
    *,
    dtype: Any = None,
    like: Any = None,
) -> torch.Tensor:
    device = like.device if like is not None else None
    if stop is None:
        return torch.arange(start, dtype=dtype, device=device)
    return torch.arange(start, stop, step, dtype=dtype, device=device)


@_TORCH.implements(ops.ones)
def ones(shape: Any, *, dtype: Any = None, like: Any = None) -> torch.Tensor:
    device = like.device if like is not None else None
    return torch.ones(shape, dtype=dtype, device=device)


@_TORCH.implements(ops.eye)
def eye(n: int, *, dtype: Any = None, like: Any = None) -> torch.Tensor:
    device = like.device if like is not None else None
    return torch.eye(n, dtype=dtype, device=device)


@_TORCH.implements(ops.diag)
def diag(value: Any) -> torch.Tensor:
    return torch.diag(value)


@_TORCH.implements(ops.tril)
def tril(value: Any, k: int = 0) -> torch.Tensor:
    return torch.tril(value, diagonal=k)


@_TORCH.implements(ops.reshape)
def reshape(value: Any, value_shape: Any) -> torch.Tensor:
    return torch.reshape(value, value_shape)


@_TORCH.implements(ops.transpose)
def transpose(value: Any, axes: Any = None) -> torch.Tensor:
    if axes is None:
        return value.permute(*reversed(range(value.dim())))
    return value.permute(*axes)


@_TORCH.implements(ops.stack)
def stack(values: Any, axis: int = 0) -> torch.Tensor:
    return torch.stack(list(values), dim=axis)


@_TORCH.implements(ops.flip)
def flip(value: Any, axis: int) -> torch.Tensor:
    return torch.flip(value, (axis,))


@_TORCH.implements(ops.roll)
def roll(value: Any, shift: int, axis: int) -> torch.Tensor:
    return torch.roll(value, shift, dims=axis)


@_TORCH.implements(ops.real)
def real(value: Any) -> torch.Tensor:
    return torch.real(value)


@_TORCH.implements(ops.log)
def log(value: Any) -> torch.Tensor:
    return torch.log(value)


@_TORCH.implements(ops.cumsum)
def cumsum(value: Any, axis: int = 0) -> torch.Tensor:
    return torch.cumsum(value, dim=axis)


@_TORCH.implements(ops.cumprod)
def cumprod(value: Any, axis: int = 0) -> torch.Tensor:
    return torch.cumprod(value, dim=axis)


@_TORCH.implements(ops.cummax)
def cummax(value: Any, axis: int = 0) -> torch.Tensor:
    return torch.cummax(value, dim=axis).values


@_TORCH.implements(ops.prod)
def prod(value: Any, axis: Any = None) -> torch.Tensor:
    return torch.prod(value) if axis is None else torch.prod(value, dim=axis)


@_TORCH.implements(ops.amax)
def amax(value: Any, axis: Any = None) -> torch.Tensor:
    return torch.amax(value) if axis is None else torch.amax(value, dim=axis)


@_TORCH.implements(ops.amin)
def amin(value: Any, axis: Any = None) -> torch.Tensor:
    return torch.amin(value) if axis is None else torch.amin(value, dim=axis)


@_TORCH.implements(ops.any)
def any(value: Any, axis: Any = None) -> torch.Tensor:
    return torch.any(value) if axis is None else torch.any(value, dim=axis)


@_TORCH.implements(ops.argmax)
def argmax(value: Any, axis: Any = None) -> torch.Tensor:
    return torch.argmax(value) if axis is None else torch.argmax(value, dim=axis)


@_TORCH.implements(ops.argsort)
def argsort(value: Any, *, descending: bool = False) -> torch.Tensor:
    return torch.argsort(value, descending=descending)


@_TORCH.implements(ops.nonzero)
def nonzero(value: Any) -> torch.Tensor:
    return torch.nonzero(value, as_tuple=False).flatten()


@_TORCH.implements(ops.matmul)
def matmul(left: Any, right: Any) -> torch.Tensor:
    return torch.matmul(left, right)


@_TORCH.implements(ops.tensordot)
def tensordot(left: Any, right: Any, axes: Any = 1) -> torch.Tensor:
    return torch.tensordot(left, right, dims=axes)


@_TORCH.implements(ops.linalg_inv)
def linalg_inv(value: Any) -> torch.Tensor:
    return torch.linalg.inv(value)


@_TORCH.implements(ops.linalg_eigvals)
def linalg_eigvals(value: Any) -> torch.Tensor:
    return torch.linalg.eigvals(value)


@_TORCH.implements(ops.fft_rfft)
def fft_rfft(value: Any, n: int | None = None) -> torch.Tensor:
    return torch.fft.rfft(value, n=n)


@_TORCH.implements(ops.fft_irfft)
def fft_irfft(value: Any, n: int | None = None) -> torch.Tensor:
    return torch.fft.irfft(value, n=n)


__all__ = ["TorchBackend"]
