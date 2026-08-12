"""Internal helper utilities for clipping operations."""

from collections.abc import Callable

from opaque.api.engine import ops
from opaque.api.engine.pytree import tree_leaves, tree_map


def normalize_to_tuple(value: int | tuple[int, ...]) -> tuple[int, ...]:
    """Normalize int or tuple of ints to tuple.

    Args:
        value: Single int or tuple of ints

    Returns:
        Tuple of ints
    """
    if isinstance(value, int):
        return (value,)
    return value


def normalize_fun_to_return_aux(fun: Callable, has_aux: bool) -> Callable:
    """Normalize function to always return (value, aux) tuple.

    Args:
        fun: Function to normalize.
        has_aux: Whether fun already returns aux.

    Returns:
        Normalized function that always returns (value, aux).
    """
    if has_aux:
        return fun
    else:
        return lambda *args, **kwargs: (fun(*args, **kwargs), ())


def batch_size_from_args(args: tuple, batch_argnums: tuple[int, ...]) -> int:
    """Determine batch size from the first batch argument.

    Handles both plain tensors and PyTree (dict/list/tuple) batch args.
    """
    first_batch_arg = args[batch_argnums[0]]
    leaves = tree_leaves(first_batch_arg)
    if not leaves:
        raise ValueError(
            f"Could not determine batch size: no tensor in batch arg at index {batch_argnums[0]}"
        )
    value_shape = ops.shape(leaves[0])
    if not value_shape:
        raise ValueError(
            f"Expected batch tensor with ndim >= 1, got 0-d tensor in batch arg at index {batch_argnums[0]}"
        )
    return value_shape[0]


def zero_grads_like(args: tuple, argnums: tuple[int, ...]):
    """Create zero gradients matching the parameter shapes at *argnums*.

    Returns a single pytree when ``len(argnums) == 1`` (matching the
    convention of ``torch.func.grad`` with a scalar argnum), or a tuple
    of pytrees otherwise.
    """
    zeros = tuple(
        tree_map(
            lambda leaf: ops.zeros_like(leaf) if ops.is_array(leaf) else leaf, args[i]
        )
        for i in argnums
    )
    if len(zeros) == 1:
        return zeros[0]
    return zeros


__all__ = [
    "batch_size_from_args",
    "normalize_fun_to_return_aux",
    "normalize_to_tuple",
    "zero_grads_like",
]
