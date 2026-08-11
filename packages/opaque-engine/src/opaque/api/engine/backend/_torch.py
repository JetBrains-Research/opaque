"""``TorchBackend`` — the PyTorch implementation of the :class:`Backend` surface.

Every method is a thin pass-through over ``torch`` / ``torch.func`` and the
existing ``opaque.api.engine`` pytree + RNG wrappers, so the compute stays
traceable under :func:`torch.func.vmap` / :func:`torch.func.grad` and the
refactor is numerically identical to the direct-torch call sites it replaces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.func import grad_and_value
from torch.func import vmap as _vmap

from opaque.api.engine.pytree import (
    tree_flatten,
    tree_flatten_with_paths,
    tree_leaves,
    tree_map,
    tree_unflatten,
)
from opaque.api.engine.random._engine import generator_from_key

if TYPE_CHECKING:
    from collections.abc import Callable

    from opaque.api.engine.random._engine import RngKey


class TorchBackend:
    """PyTorch-backed implementation of the five-primitive backend surface."""

    name = "torch"

    # --- autodiff ---
    def value_and_grad(
        self,
        fn: Callable[..., Any],
        argnums: int | tuple[int, ...] = 0,
        has_aux: bool = False,
    ) -> Callable[..., Any]:
        """Delegate to :func:`torch.func.grad_and_value` (returns ``(grad, value)``)."""
        return grad_and_value(fn, argnums=argnums, has_aux=has_aux)

    # --- vectorization ---
    def vmap(
        self,
        fn: Callable[..., Any],
        in_axes: Any = 0,
        out_axes: Any = 0,
        randomness: str = "error",
    ) -> Callable[..., Any]:
        """Delegate to :func:`torch.func.vmap`."""
        return _vmap(fn, in_dims=in_axes, out_dims=out_axes, randomness=randomness)

    # --- pytree ---
    def tree_map(self, fn: Callable[..., Any], *trees: Any) -> Any:
        return tree_map(fn, *trees)

    def tree_flatten(self, tree: Any) -> tuple[list[Any], Any]:
        return tree_flatten(tree)

    def tree_flatten_with_paths(self, tree: Any) -> tuple[list[Any], list[Any], Any]:
        return tree_flatten_with_paths(tree)

    def tree_unflatten(self, treedef: Any, leaves: list[Any]) -> Any:
        return tree_unflatten(treedef, leaves)

    def tree_leaves(self, tree: Any) -> list[Any]:
        return tree_leaves(tree)

    # --- array math (elementwise + reduction + dtype helpers) ---
    def is_array(self, x: Any) -> bool:
        return isinstance(x, torch.Tensor)

    def is_floating(self, x: Any) -> bool:
        if isinstance(x, torch.dtype):
            return torch.is_floating_point(torch.empty((), dtype=x))
        return torch.is_floating_point(x)

    def sqrt(self, x: Any) -> Any:
        return torch.sqrt(x)

    def square(self, x: Any) -> Any:
        return torch.square(x)

    def sum(self, x: Any, axis: Any = None, dtype: Any = None) -> Any:
        if axis is None:
            return torch.sum(x, dtype=dtype)
        return torch.sum(x, dim=axis, dtype=dtype)

    def minimum(self, a: Any, b: Any) -> Any:
        return torch.minimum(a, b)

    def maximum(self, a: Any, b: Any) -> Any:
        return torch.maximum(a, b)

    def where(self, cond: Any, a: Any, b: Any) -> Any:
        return torch.where(cond, a, b)

    def isfinite(self, x: Any) -> Any:
        return torch.isfinite(x)

    def nan_to_num(self, x: Any) -> Any:
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def clamp(self, x: Any, lo: Any = None, hi: Any = None) -> Any:
        return torch.clamp(x, min=lo, max=hi)

    def zeros_like(self, x: Any) -> Any:
        return torch.zeros_like(x)

    def concatenate(self, xs: Any, axis: int = 0) -> Any:
        return torch.cat(tuple(xs), dim=axis)

    def astype(self, x: Any, dtype: Any) -> Any:
        return x.to(dtype)

    def scalar(self, value: Any, *, dtype: Any = None, like: Any = None) -> Any:
        device = like.device if like is not None else None
        return torch.tensor(value, dtype=dtype, device=device)

    def promote_dtype(self, a: Any, b: Any) -> Any:
        return torch.promote_types(a, b)

    # --- rng (surface complete; consumer rewrite deferred) ---
    def generator(self, key: RngKey) -> torch.Generator:
        return generator_from_key(key)

    def normal(self, shape: Any, *, dtype: Any, generator: Any) -> torch.Tensor:
        return torch.randn(shape, dtype=dtype, generator=generator)


__all__ = ["TorchBackend"]
