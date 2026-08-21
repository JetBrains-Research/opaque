"""Torch implementations of the portable core primitives."""

from __future__ import annotations

import functools
import math
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


@_TORCH.implements(ops.float64)
def float64() -> torch.dtype:
    return torch.float64


@_TORCH.implements(ops.boolean)
def boolean() -> torch.dtype:
    return torch.bool


@_TORCH.implements(ops.real_dtype)
def real_dtype(value: Any) -> torch.dtype:
    value_dtype = value if isinstance(value, torch.dtype) else value.dtype
    return torch.empty((), dtype=value_dtype).real.dtype


def _placement(dtype: Any, like: Any) -> tuple[Any, Any]:
    """Resolve ``(dtype, device)`` for a creation op.

    ``like`` supplies both, matching ``zeros_like`` and the keyed ``normal``
    draw; an explicit ``dtype`` overrides its dtype half.  Taking only the
    device would put a constant beside an ``float64`` leaf at the default
    ``float32`` without saying so.
    """
    if like is None:
        return dtype, None
    return (dtype if dtype is not None else like.dtype), like.device


@_TORCH.implements(ops.scalar)
def scalar(value: Any, *, dtype: Any = None, like: Any = None) -> torch.Tensor:
    resolved_dtype, device = _placement(dtype, like)
    return torch.tensor(value, dtype=resolved_dtype, device=device)


@_TORCH.implements(ops.zeros)
def zeros(shape: Any, *, dtype: Any = None, like: Any = None) -> torch.Tensor:
    resolved_dtype, device = _placement(dtype, like)
    return torch.zeros(shape, dtype=resolved_dtype, device=device)


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


def _as_dtype(value: Any) -> torch.dtype:
    """Accept an array or a dtype, as the dtype predicates already do."""
    return value.dtype if isinstance(value, torch.Tensor) else value


@_TORCH.implements(ops.finfo_eps)
def finfo_eps(value_dtype: Any) -> float:
    return float(torch.finfo(_as_dtype(value_dtype)).eps)


@_TORCH.implements(ops.finfo_smallest_normal)
def finfo_smallest_normal(value_dtype: Any) -> float:
    return float(torch.finfo(_as_dtype(value_dtype)).smallest_normal)


@_TORCH.implements(ops.to_host)
def to_host(value: Any) -> Any:
    # ``.copy()`` is load-bearing, not defensive. ``Tensor.numpy()`` shares
    # storage, and ``.cpu()`` is a no-op on a tensor already there — so
    # without it the "copy" this primitive promises is an alias on CPU and a
    # real copy on CUDA. A caller normalizing scores in place would silently
    # write back into the graph on one device and not the other.
    return value.detach().cpu().numpy().copy()


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


@_TORCH.implements(ops.amin)
def amin(value: Any, axis: Any = None) -> torch.Tensor:
    return torch.amin(value) if axis is None else torch.amin(value, dim=axis)


@_TORCH.implements(ops.amax)
def amax(value: Any, axis: Any = None) -> torch.Tensor:
    return torch.amax(value) if axis is None else torch.amax(value, dim=axis)


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
def nan_to_num(
    value: Any,
    *,
    nan: float = 0.0,
    posinf: float = 0.0,
    neginf: float = 0.0,
) -> torch.Tensor:
    return torch.nan_to_num(value, nan=nan, posinf=posinf, neginf=neginf)


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
    return torch.promote_types(_as_dtype(first), _as_dtype(second))


def _under_differentiating_transform() -> bool:
    """Return whether an enclosing ``grad``, ``vjp``, or ``jvp`` is active."""
    if torch.compiler.is_compiling():
        # The interpreter stack is not observable while tracing; assume the
        # graph is wanted rather than silently dropping it.
        return True
    try:
        from torch._C._functorch import TransformType, get_interpreter_stack

        differentiating = (TransformType.Grad, TransformType.Jvp)
        stack = get_interpreter_stack() or ()
        return any(interpreter.key() in differentiating for interpreter in stack)
    except Exception:  # pragma: no cover - private API moved/unavailable
        return True


@_TORCH.implements(autodiff._grad_and_value_transform)
def grad_and_value(
    fn: Any,
    argnums: Any = 0,
    has_aux: bool = False,
    values_only: bool = False,
) -> Any:
    transformed = _grad_and_value(fn, argnums, has_aux)
    if not values_only:
        return transformed

    # ``torch.func``'s internal backward runs with ``create_graph=True``, so
    # it builds a differentiable graph the caller has declared it will not
    # use. Skip that work — but only when nothing enclosing differentiates
    # this result, which is a per-invocation question.
    @functools.wraps(transformed)
    def values_only_transform(*args: Any, **kwargs: Any) -> Any:
        if _under_differentiating_transform():
            return transformed(*args, **kwargs)
        with torch.no_grad():
            return transformed(*args, **kwargs)

    return values_only_transform


@_TORCH.implements(autodiff._vmap_transform)
def vmap(fn: Any, in_axes: Any = 0, out_axes: Any = 0, randomness: str = "same") -> Any:
    return _vmap(fn, in_dims=in_axes, out_dims=out_axes, randomness=randomness)


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


_SQ_NORM_ACCUM_DTYPE = torch.float64
"""Dtype the cross-leaf squared-norm accumulator runs in.

The per-leaf squaring still happens in the caller's ``compute_dtype``.
"""

_REDUCTION_BLOCK = 2048
"""Width of one block in the two-stage leaf reduction.

Deliberately a compile-time constant rather than ``isqrt(numel)``.  The
balanced split is the tightest bound, but ``math.isqrt`` on a symbolic shape
makes ``torch.compile(fullgraph=True)`` fail outright (and the default mode
fall back to a graph break) for every leaf past the threshold below.  Since
``clipped_grad`` is on the hot path, breaking compilation there breaks it
everywhere, which costs more than the slack a fixed block gives up.

At 2048 the two stages stay balanced for the multi-million-element leaves that
motivate the guard — the bound matches the ``isqrt`` one to within a term at
``numel ~ 4e6``.  Smaller leaves get a looser bound than ``isqrt`` would give,
but still a far tighter one than the flat reduction they would otherwise use.
"""

_BLOCKED_REDUCTION_MIN = 4096
"""Element count above which a leaf's sum of squares is reduced in two stages.

A flat reduction admits ``numel`` sequential roundings; splitting it into
blocks of ``_REDUCTION_BLOCK`` admits ``_REDUCTION_BLOCK + numel /
_REDUCTION_BLOCK``, which is what keeps the guard affordable on the float32
accumulator MPS is limited to.  Below the threshold the flat term is already
negligible and the extra kernel is not worth it.
"""


def _sq_accum_dtype(leaves: list[torch.Tensor]) -> torch.dtype:
    """Dtype for the cross-leaf squared-norm accumulator.

    MPS has no float64, so it accumulates in float32 and the guard widens to
    match; every other backend gets the free float64 scalar.
    """
    if any(leaf.device.type == "mps" for leaf in leaves):
        return torch.float32
    return _SQ_NORM_ACCUM_DTYPE


def _blocked_reduction_terms(n: int) -> int:
    """Worst-case sequential additions for a two-stage reduction of ``n``.

    ``_REDUCTION_BLOCK`` additions inside the widest block, then one per block.
    Integer arithmetic only, so it survives tracing on a symbolic ``n``.
    """
    block = _REDUCTION_BLOCK
    return block + (n + block - 1) // block


def _reduction_terms(leaves: list[torch.Tensor]) -> int:
    """Worst-case sequential additions inside a single leaf's reduction.

    Written as a loop rather than ``max(..., default=0)`` over a generator,
    which ``torch.compile`` rejects.
    """
    widest = 0
    for leaf in leaves:
        widest = max(widest, math.prod(leaf.shape))
    if widest <= 1:
        return 0
    if widest > _BLOCKED_REDUCTION_MIN:
        return _blocked_reduction_terms(widest)
    return widest


def _leaf_sq_sum(
    leaf: torch.Tensor, acc_dtype: torch.dtype, sq_dtype: torch.dtype
) -> torch.Tensor:
    """Sum of squares of one leaf, reduced in ``sq_dtype``.

    Complex leaves contribute ``real^2 + imag^2``; casting them to a real
    accumulator would drop the imaginary part and under-read the norm.
    """
    if torch.is_complex(leaf):
        real = leaf.real.to(acc_dtype)
        imag = leaf.imag.to(acc_dtype)
        sq = real * real + imag * imag
    else:
        x = leaf.to(acc_dtype)
        sq = x * x

    flat = sq.reshape(-1)
    n = flat.shape[0]
    if n <= _BLOCKED_REDUCTION_MIN:
        return flat.sum(dtype=sq_dtype)

    block = _REDUCTION_BLOCK
    main = (n // block) * block
    total = (
        flat[:main].reshape(-1, block).sum(dim=-1, dtype=sq_dtype).sum(dtype=sq_dtype)
    )
    if main < n:
        total = total + flat[main:].sum(dtype=sq_dtype)
    return total


@_TORCH.implements(pytree._squared_l2_norms)
def squared_l2_norms(
    leaves: list[torch.Tensor],
    groups: list[str] | None,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    sq_dtype = _sq_accum_dtype(leaves)
    total: torch.Tensor | None = None
    grouped: dict[str, torch.Tensor] = {}
    for index, leaf in enumerate(leaves):
        sq = _leaf_sq_sum(leaf, dtype, sq_dtype)
        total = sq if total is None else total + sq
        if groups is not None:
            group = groups[index]
            grouped[group] = grouped[group] + sq if group in grouped else sq
    if total is None:
        total = torch.zeros((), dtype=sq_dtype)
    return total, grouped


@_TORCH.implements(pytree._squared_l2_norm_roundoff)
def squared_l2_norm_roundoff(
    leaves: list[torch.Tensor],
    *,
    dtype: torch.dtype,
) -> float:
    """Relative error bound on the computed norm.

    Three sources: squaring each element in ``dtype``, the reduction inside
    the widest leaf (see :func:`_reduction_terms`), and the cross-leaf
    accumulation.  Halved because the error is carried through a square root.
    """
    sq_dtype = _sq_accum_dtype(leaves)
    u_acc = torch.finfo(dtype).eps / 2.0 if dtype.is_floating_point else 0.0
    u_sq = torch.finfo(sq_dtype).eps / 2.0 if sq_dtype.is_floating_point else 0.0
    return (u_acc + (max(len(leaves), 1) + _reduction_terms(leaves)) * u_sq) / 2.0


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


__all__ = ["TorchBackend"]
