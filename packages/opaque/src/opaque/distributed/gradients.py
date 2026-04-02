"""Gradient aggregation for distributed DP training.

This module provides functions to aggregate PyTrees of gradients across devices:
- reduce_pytree: Generic reduction returning a new reduced PyTree
- reduce_pytree_: Generic in-place reduction of PyTree tensors (sum, mean, max, min)
- sum_gradients: DP-specific helper that sums clipped gradients across devices

These work with Opaque's functional API which produces clipped+summed gradients
per device. The aggregation happens AFTER per-example clipping on each device.
"""

from typing import Any

import torch

from opaque.utils.pytree import tree_map

from . import all_reduce_ as all_reduce_tensor
from . import is_distributed

__all__ = [
    "reduce_pytree",
    "reduce_pytree_",
    "sum_gradients",
    "sum_gradients_",
]


def reduce_pytree_(
    pytree: Any,
    op: str = "sum",
) -> None:
    """Reduce a PyTree of tensors across all processes (in-place).

    This applies all_reduce to each tensor in the PyTree independently.
    The operation is in-place: tensors in the input PyTree are modified.

    Args:
        pytree: PyTree (nested dict/list/tuple) of tensors to reduce.
        op: Reduction operation. One of: "sum", "mean", "max", "min", "product".
            Default: "sum".

    Returns:
        None. Tensor leaves in ``pytree`` are reduced in-place.

    Example:
        >>> import torch
        >>> import torch.distributed as dist
        >>> from opaque.distributed import reduce_pytree_
        >>>
        >>> # Initialize distributed (rank 0 of 2)
        >>> dist.init_process_group(backend='nccl', rank=0, world_size=2)
        >>>
        >>> # Each device has different tensors
        >>> params = {
        ...     "weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        ...     "bias": torch.tensor([0.5, 1.0]),
        ... }
        >>>
        >>> # Sum across all devices
        >>> reduce_pytree_(params, op="sum")
        >>> # params["weight"] is now [[2.0, 4.0], [6.0, 8.0]] (sum across devices)

    Notes:
        - If distributed is not initialized, this is a no-op
        - Operates in-place for memory efficiency
        - Generic reduction - works with any PyTree, not just gradients
        - Always blocking (no async execution)
    """
    if not is_distributed():
        return

    def reduce_leaf(tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(tensor, torch.Tensor):
            all_reduce_tensor(tensor, op=op)
        return tensor

    # Apply all_reduce to each tensor in the PyTree
    tree_map(reduce_leaf, pytree)


def reduce_pytree(
    pytree: Any,
    op: str = "sum",
) -> Any:
    """Reduce a PyTree of tensors across all processes and return a new tree.

    This is the functional counterpart to ``reduce_pytree_``.

    Args:
        pytree: PyTree (nested dict/list/tuple) of tensors to reduce.
        op: Reduction operation. One of: "sum", "mean", "max", "min", "product".
            Default: "sum".

    Returns:
        A new PyTree with reduced tensor leaves. The input ``pytree`` is unchanged.
    """

    def clone_leaf(leaf: Any) -> Any:
        if isinstance(leaf, torch.Tensor):
            return leaf.clone()
        return leaf

    reduced = tree_map(clone_leaf, pytree)
    reduce_pytree_(reduced, op=op)
    return reduced


def sum_gradients_(gradients: Any) -> None:
    """Sum clipped gradients across all devices (DP-specific helper, in-place).

    This is a convenience wrapper around ``reduce_pytree_(op="sum")`` specifically
    for differential privacy training where we need to sum clipped gradients
    from all devices before adding noise.

    Args:
        gradients: PyTree of clipped gradients to sum.

    Returns:
        None. Gradient tensors are summed in-place.

    Example:
        >>> import torch
        >>> from opaque.distributed import sum_gradients_
        >>> from opaque.clipping import clipped_grad
        >>> from opaque.noise import gaussian_noise
        >>>
        >>> # Each device computes clipped gradients on local batch
        >>> # (Poisson sampling: different batch sizes!)
        >>> loss_fn = lambda params, x, y: ((x @ params - y) ** 2).mean()
        >>> grad_fn = clipped_grad(loss_fn, clipping_norm=1.0, batch_argnums=(1, 2))
        >>>
        >>> # Rank 0: 32 examples, Rank 1: 28 examples
        >>> grads = grad_fn(params, batch_x, batch_y)  # Sum of B clipped grads
        >>>
        >>> # Sum across devices: total 60 examples
        >>> sum_gradients_(grads)
        >>>
        >>> # Add noise scaled to clipping_norm (NOT dependent on batch size!)
        >>> noisy_grads = gaussian_noise(grads, sigma=1.1)

    Notes:
        - Essential for DP-SGD with Poisson sampling across devices
        - Clipped gradients have sensitivity C (independent of batch size)
        - Noise is added AFTER summing (scaled to C, not B*C)
        - If distributed is not initialized, this is a no-op
    """
    reduce_pytree_(gradients, op="sum")


def sum_gradients(gradients: Any) -> Any:
    """Sum clipped gradients across all devices and return a reduced copy.

    This is the functional counterpart to ``sum_gradients_``.

    Args:
        gradients: PyTree of clipped gradients to sum.

    Returns:
        A new PyTree with summed gradient tensors. Input ``gradients`` is unchanged.
    """
    return reduce_pytree(gradients, op="sum")
