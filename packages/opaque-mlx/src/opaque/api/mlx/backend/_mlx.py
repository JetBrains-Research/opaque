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
        """Vectorize with MLX, rejecting policies it cannot implement."""
        if randomness != "error":
            raise ValueError(
                "MLX vmap supports only Opaque's randomness='error' mode; "
                f"got {randomness!r}."
            )
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
        return [leaf for _, leaf in mx_utils.tree_flatten(tree) if self.is_array(leaf)]

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
        dtype = x if isinstance(x, mx.Dtype) else x.dtype
        complex128 = getattr(mx, "complex128", mx.complex64)
        return dtype == mx.complex64 or dtype == complex128

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

    def slice_array(self, value: Any, slices: Any) -> Any:
        return value[slices]

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


def _normal(rng_key: RngKey, shape: Any, *, dtype: Any = None, like: Any = None) -> Any:
    resolved_dtype = dtype or (like.dtype if like is not None else mx.float32)
    return mx.random.normal(
        shape, dtype=resolved_dtype, key=mx.random.key(rng_key.seed)
    )


def register_core_primitives() -> None:
    """Idempotently register MLX implementations for the portable core."""
    from opaque.api.engine import autodiff, ops, pytree
    from opaque.api.engine.random import _engine as random_engine

    backend = MlxBackend()

    def dtype(value: Any) -> Any:
        return value.dtype

    def shape(value: Any) -> tuple[int, ...]:
        return tuple(value.shape)

    def real_dtype(value: Any) -> Any:
        value_dtype = value.dtype if backend.is_array(value) else value
        return mx.real(mx.zeros((), dtype=value_dtype)).dtype

    def zeros(value_shape: Any, *, dtype: Any = None, like: Any = None) -> Any:
        del like
        return mx.zeros(value_shape, dtype=dtype)

    def ones_like(value: Any) -> Any:
        return mx.ones_like(value)

    def clone(value: Any) -> Any:
        return mx.array(value)

    def detach(value: Any) -> Any:
        return value

    def transfer(value: Any, *args: Any, **kwargs: Any) -> Any:
        dtype = kwargs.get("dtype")
        if dtype is not None:
            return value.astype(dtype)
        if args and isinstance(args[0], mx.Dtype):
            return value.astype(args[0])
        return value

    def scalar_item(value: Any) -> Any:
        return value.item()

    def abs(value: Any) -> Any:
        return mx.abs(value)

    def add(left: Any, right: Any) -> Any:
        return mx.add(left, right)

    def subtract(left: Any, right: Any) -> Any:
        return mx.subtract(left, right)

    def multiply(left: Any, right: Any) -> Any:
        return mx.multiply(left, right)

    def divide(left: Any, right: Any) -> Any:
        return mx.divide(left, right)

    def all(value: Any, axis: Any = None) -> Any:
        return mx.all(value, axis=axis)

    def greater(left: Any, right: Any) -> Any:
        return mx.greater(left, right)

    def tree_structure(tree: Any) -> Any:
        return optree.tree_structure(tree)

    registrations = (
        (ops.is_array, backend.is_array),
        (ops.dtype, dtype),
        (ops.shape, shape),
        (ops.is_floating, backend.is_floating),
        (ops.is_low_precision, backend.is_low_precision),
        (ops.is_complex, backend.is_complex),
        (ops.float32, lambda: mx.float32),
        (ops.real_dtype, real_dtype),
        (ops.scalar, backend.scalar),
        (ops.zeros, zeros),
        (ops.zeros_like, backend.zeros_like),
        (ops.ones_like, ones_like),
        (ops.astype, backend.astype),
        (ops.clone, clone),
        (ops.detach, detach),
        (ops.transfer, transfer),
        (ops.scalar_item, scalar_item),
        (ops.sqrt, backend.sqrt),
        (ops.square, backend.square),
        (ops.abs, abs),
        (ops.add, add),
        (ops.subtract, subtract),
        (ops.multiply, multiply),
        (ops.divide, divide),
        (ops.sum, backend.sum),
        (ops.greater, greater),
        (ops.minimum, backend.minimum),
        (ops.maximum, backend.maximum),
        (ops.where, backend.where),
        (ops.isfinite, backend.isfinite),
        (ops.all, all),
        (ops.nan_to_num, backend.nan_to_num),
        (ops.clamp, backend.clamp),
        (ops.concatenate, backend.concatenate),
        (ops.slice_array, backend.slice_array),
        (ops.promote_dtype, backend.promote_dtype),
        (autodiff.grad_and_value, backend.value_and_grad),
        (autodiff.vmap, backend.vmap),
        (pytree._tree_map, backend.tree_map),
        (pytree._tree_flatten, backend.tree_flatten),
        (pytree._tree_flatten_with_paths, backend.tree_flatten_with_paths),
        (pytree._tree_unflatten, backend.tree_unflatten),
        (pytree._tree_leaves, backend.tree_leaves),
        (pytree._tree_structure, tree_structure),
        (random_engine._normal, _normal),
    )
    for primitive, implementation in registrations:
        if not primitive.supports("mlx"):
            primitive.register("mlx", implementation)


__all__ = ["MlxBackend", "register_core_primitives"]
