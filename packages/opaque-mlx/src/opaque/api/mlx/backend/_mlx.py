"""MLX implementation of Opaque's neutral compute backend protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optree

import mlx.core as mx
import mlx.utils as mx_utils
from opaque.api.engine.pytree import param_path

if TYPE_CHECKING:
    from collections.abc import Callable

    from opaque.api.engine.random._engine import RngKey


class MlxBackend:
    """MLX-backed implementation of the five-primitive backend surface."""

    name = "mlx"
    float32 = mx.float32

    # --- autodiff ---
    def value_and_grad(
        self,
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

    # --- vectorization ---
    def vmap(
        self,
        fn: Callable[..., Any],
        in_axes: Any = 0,
        out_axes: Any = 0,
        randomness: str = "error",
    ) -> Callable[..., Any]:
        """Vectorize with MLX; its API has no separate randomness policy."""
        del randomness
        return mx.vmap(fn, in_axes=in_axes, out_axes=out_axes)

    # --- pytree ---
    def tree_map(self, fn: Callable[..., Any], *trees: Any) -> Any:
        return mx_utils.tree_map(fn, *trees)

    def tree_flatten(self, tree: Any) -> tuple[list[Any], Any]:
        # MLX's dotted flattening loses the distinction between a flat dotted
        # key and nested dictionaries. optree supplies the structural treedef
        # required by the engine's stable ParamPath contract.
        leaves, treedef = optree.tree_flatten(tree)
        return list(leaves), treedef

    def tree_flatten_with_paths(self, tree: Any) -> tuple[list[Any], list[Any], Any]:
        paths, leaves, treedef = optree.tree_flatten_with_path(tree)
        return [param_path(path) for path in paths], list(leaves), treedef

    def tree_unflatten(self, treedef: Any, leaves: list[Any]) -> Any:
        return optree.tree_unflatten(treedef, leaves)

    def tree_leaves(self, tree: Any) -> list[Any]:
        return [
            leaf for _, leaf in mx_utils.tree_flatten(tree) if self.is_array(leaf)
        ]

    # --- array math (elementwise + reduction + dtype helpers) ---
    def is_array(self, x: Any) -> bool:
        return isinstance(x, mx.array)

    def is_floating(self, x: Any) -> bool:
        dtype = x if isinstance(x, mx.Dtype) else x.dtype
        return mx.issubdtype(dtype, mx.floating)

    def is_low_precision(self, x: Any) -> bool:
        dtype = x if isinstance(x, mx.Dtype) else x.dtype
        return dtype in (mx.float16, mx.bfloat16)

    def is_complex(self, x: Any) -> bool:
        del x
        return False

    def sqrt(self, x: Any) -> Any:
        return mx.sqrt(x)

    def square(self, x: Any) -> Any:
        return mx.square(x)

    def sum(self, x: Any, axis: Any = None, dtype: Any = None) -> Any:
        if dtype is not None:
            x = x.astype(dtype)
        return mx.sum(x, axis=axis)

    def minimum(self, a: Any, b: Any) -> Any:
        return mx.minimum(a, b)

    def maximum(self, a: Any, b: Any) -> Any:
        return mx.maximum(a, b)

    def where(self, cond: Any, a: Any, b: Any) -> Any:
        return mx.where(cond, a, b)

    def isfinite(self, x: Any) -> Any:
        return mx.isfinite(x)

    def nan_to_num(self, x: Any) -> Any:
        return mx.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def clamp(self, x: Any, lo: Any = None, hi: Any = None) -> Any:
        if lo is not None:
            x = mx.maximum(x, lo)
        if hi is not None:
            x = mx.minimum(x, hi)
        return x

    def zeros_like(self, x: Any) -> Any:
        return mx.zeros_like(x)

    def concatenate(self, xs: Any, axis: int = 0) -> Any:
        return mx.concatenate(tuple(xs), axis=axis)

    def astype(self, x: Any, dtype: Any) -> Any:
        return x.astype(dtype)

    def scalar(self, value: Any, *, dtype: Any = None, like: Any = None) -> Any:
        del like
        return mx.array(value, dtype=dtype)

    def promote_dtype(self, a: Any, b: Any) -> Any:
        return mx.result_type(a, b)

    # --- rng ---
    def generator(self, key: RngKey) -> Any:
        return mx.random.key(key.seed)

    def normal(self, shape: Any, *, dtype: Any, generator: Any) -> Any:
        return mx.random.normal(shape, dtype=dtype, key=generator)


__all__ = ["MlxBackend"]
