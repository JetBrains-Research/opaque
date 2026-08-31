"""Internal helper utilities for clipping operations."""

from collections.abc import Callable

import torch

from opaque.api.engine.pytree import tree_map
from opaque.exceptions import ConfigurationError


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
    if isinstance(first_batch_arg, torch.Tensor):
        return first_batch_arg.shape[0]

    def _first_tensor(pytree):
        if isinstance(pytree, torch.Tensor):
            return pytree
        if isinstance(pytree, dict):
            for v in pytree.values():
                t = _first_tensor(v)
                if t is not None:
                    return t
        elif isinstance(pytree, (list, tuple)):
            for v in pytree:
                t = _first_tensor(v)
                if t is not None:
                    return t
        return None

    tensor = _first_tensor(first_batch_arg)
    if tensor is None:
        raise ConfigurationError(
            *(
                f"Could not determine batch size: no tensor in batch arg at index {batch_argnums[0]}",
            )
        )
    if tensor.ndim < 1:
        raise ConfigurationError(
            *(
                f"Expected batch tensor with ndim >= 1, got 0-d tensor in batch arg at index {batch_argnums[0]}",
            )
        )
    return tensor.shape[0]


def zero_grads_like(args: tuple, argnums: tuple[int, ...]):
    """Create zero gradients matching the parameter shapes at *argnums*.

    Returns a single pytree when ``len(argnums) == 1`` (matching the
    convention of ``torch.func.grad`` with a scalar argnum), or a tuple
    of pytrees otherwise.
    """
    zeros = tuple(tree_map(torch.zeros_like, args[i]) for i in argnums)
    if len(zeros) == 1:
        return zeros[0]
    return zeros


def empty_clipped_grads_like(
    args: tuple,
    argnums: tuple[int, ...],
    pre_clipping_transform: Callable = lambda x: x,
    dtype: torch.dtype | None = None,
):
    """Zero gradients matching the structure/dtype of a non-empty step.

    Mirrors the non-empty clipping pipeline so empty-batch short-circuits
    return step-stable outputs: ``pre_clipping_transform`` is applied to the
    zero gradient pytree first (it may change the tree structure), then
    tensor leaves are cast to the configured output ``dtype`` (``None`` keeps
    the input dtype).

    Args:
        args: Positional arguments of the wrapped loss function.
        argnums: Which argument(s) the zeros correspond to.
        pre_clipping_transform: Transform applied to the gradient pytree,
            identical to the one used on non-empty steps.
        dtype: Optional output dtype to cast leaves to, matching the
            ``dtype`` argument of ``clipped_grad`` / ``clipped_fun``.

    Returns:
        Zero-valued pytree (or tuple of pytrees, following the
        ``zero_grads_like`` convention) with the same structure and dtypes
        a non-empty step would produce.
    """
    zeros = zero_grads_like(args, argnums)
    transformed = pre_clipping_transform(zeros)
    if dtype is None:
        return transformed
    return tree_map(
        lambda x: x.to(dtype) if isinstance(x, torch.Tensor) else x, transformed
    )


__all__ = [
    "batch_size_from_args",
    "empty_clipped_grads_like",
    "normalize_fun_to_return_aux",
    "normalize_to_tuple",
    "zero_grads_like",
]
