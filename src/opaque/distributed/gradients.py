"""Gradient aggregation for distributed DP training.

This module provides functions to aggregate PyTrees of gradients across devices:
- all_reduce_gradients: Sum gradients across all devices
- average_gradients: Average gradients across all devices

These work with Opaque's functional API which produces clipped+summed gradients
per device. The aggregation happens AFTER per-example clipping on each device.
"""

from typing import Any

import torch

from opaque.utils.pytree import tree_map

from . import all_reduce as all_reduce_tensor
from . import get_world_size, is_initialized

__all__ = [
    "all_reduce_gradients",
    "average_gradients",
]


def all_reduce_gradients(
    gradients: Any,
    op: str = "sum",
    async_op: bool = False,
) -> tuple[Any, list[Any] | None]:
    """All-reduce a PyTree of gradients across all processes (in-place).

    This applies all_reduce to each tensor in the PyTree independently.
    The operation is in-place: tensors in the input PyTree are modified.

    Args:
        gradients: PyTree (nested dict/list/tuple) of tensors to reduce.
        op: Reduction operation. One of: "sum", "mean", "max", "min", "product".
            Default: "sum".
        async_op: If True, return work handles for asynchronous operation.
            Default: False (blocking).

    Returns:
        Tuple of (gradients, work_handles):
            - gradients: Same PyTree with reduced tensors (modified in-place)
            - work_handles: List of dist.Work handles if async_op=True, else None

    Raises:
        RuntimeError: If distributed is not initialized.

    Example:
        >>> import torch
        >>> import torch.distributed as dist
        >>> import opaque.distributed.gradients as dist_grads
        >>>
        >>> # Initialize distributed (rank 0 of 2)
        >>> dist.init_process_group(backend='nccl', rank=0, world_size=2)
        >>>
        >>> # Each device has different gradients (after clipping)
        >>> grads = {
        ...     "weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        ...     "bias": torch.tensor([0.5, 1.0]),
        ... }
        >>>
        >>> # Sum across all devices
        >>> grads, _ = dist_grads.all_reduce_gradients(grads, op="sum")
        >>> # grads["weight"] is now [[2.0, 4.0], [6.0, 8.0]] (sum of rank 0 and 1)

    Notes:
        - If distributed is not initialized, returns input unchanged
        - For DP training, typically use op="sum" then add noise to summed gradients
        - Operates in-place for memory efficiency
    """
    if not is_initialized():
        return gradients, None

    # Collect work handles if async
    work_handles: list[Any] | None = [] if async_op else None

    def reduce_leaf(tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(tensor, torch.Tensor):
            work = all_reduce_tensor(tensor, op=op, async_op=async_op)
            if async_op and work_handles is not None:
                work_handles.append(work)
        return tensor

    # Apply all_reduce to each tensor in the PyTree
    tree_map(reduce_leaf, gradients)

    return gradients, work_handles


def average_gradients(
    gradients: Any,
    world_size: int | None = None,
) -> Any:
    """Average a PyTree of gradients across all processes.

    This is equivalent to all_reduce_gradients(op="sum") followed by division
    by world_size. Useful when you want the mean gradient instead of sum.

    Args:
        gradients: PyTree (nested dict/list/tuple) of tensors to average.
        world_size: Number of processes. If None, uses get_world_size().
            Default: None.

    Returns:
        gradients: PyTree with averaged tensors (modified in-place).

    Example:
        >>> import torch
        >>> import torch.distributed as dist
        >>> import opaque.distributed.gradients as dist_grads
        >>>
        >>> # Initialize distributed (rank 0 of 2)
        >>> dist.init_process_group(backend='nccl', rank=0, world_size=2)
        >>>
        >>> # Each device has 32 examples, clipped and summed locally
        >>> # Rank 0: sum of 32 clipped gradients
        >>> # Rank 1: sum of 32 clipped gradients
        >>> grads = {
        ...     "weight": torch.randn(10, 5),
        ...     "bias": torch.randn(5),
        ... }
        >>>
        >>> # Average across both devices (total 64 examples)
        >>> grads = dist_grads.average_gradients(grads)
        >>> # grads now contains average of 64 clipped gradients

    Notes:
        - For DP training with equal batch sizes per device, averaging vs summing
          only changes the effective learning rate
        - Operates in-place for memory efficiency
        - If distributed is not initialized, returns input unchanged
    """
    if not is_initialized():
        return gradients

    if world_size is None:
        world_size = get_world_size()

    # Sum across devices
    gradients, _ = all_reduce_gradients(gradients, op="sum", async_op=False)

    # Divide by world_size
    def average_leaf(tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(tensor, torch.Tensor):
            tensor.div_(world_size)
        return tensor

    tree_map(average_leaf, gradients)

    return gradients
