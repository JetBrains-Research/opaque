"""Gradient aggregation for distributed DP training.

This module provides functions to aggregate PyTrees of gradients across devices:
- reduce_pytree: Generic reduction of PyTree tensors (sum, mean, max, min)
- sum_gradients: DP-specific helper that sums clipped gradients across devices

These work with Opaque's functional API which produces clipped+summed gradients
per device. The aggregation happens AFTER per-example clipping on each device.
"""

from typing import Any

import torch

from opaque.utils.pytree import tree_map

from . import all_reduce as all_reduce_tensor
from . import is_distributed

__all__ = [
    "reduce_pytree",
    "sum_gradients",
]


def reduce_pytree(
    pytree: Any,
    op: str = "sum",
    async_op: bool = False,
) -> tuple[Any, list[Any] | None]:
    """Reduce a PyTree of tensors across all processes (in-place).

    This applies all_reduce to each tensor in the PyTree independently.
    The operation is in-place: tensors in the input PyTree are modified.

    Args:
        pytree: PyTree (nested dict/list/tuple) of tensors to reduce.
        op: Reduction operation. One of: "sum", "mean", "max", "min", "product".
            Default: "sum".
        async_op: If True, return work handles for asynchronous operation.
            Default: False (blocking).

    Returns:
        Tuple of (pytree, work_handles):
            - pytree: Same PyTree with reduced tensors (modified in-place)
            - work_handles: List of dist.Work handles if async_op=True, else None

    Example:
        >>> import torch
        >>> import torch.distributed as dist
        >>> from opaque.distributed import reduce_pytree
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
        >>> params, _ = reduce_pytree(params, op="sum")
        >>> # params["weight"] is now [[2.0, 4.0], [6.0, 8.0]] (sum across devices)

    Notes:
        - If distributed is not initialized, returns input unchanged
        - Operates in-place for memory efficiency
        - Generic reduction - works with any PyTree, not just gradients
    """
    if not is_distributed():
        return pytree, None

    # Collect work handles if async
    work_handles: list[Any] | None = [] if async_op else None

    def reduce_leaf(tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(tensor, torch.Tensor):
            work = all_reduce_tensor(tensor, op=op, async_op=async_op)
            if async_op and work_handles is not None:
                work_handles.append(work)
        return tensor

    # Apply all_reduce to each tensor in the PyTree
    tree_map(reduce_leaf, pytree)

    return pytree, work_handles


def sum_gradients(gradients: Any) -> Any:
    """Sum clipped gradients across all devices (DP-specific helper).

    This is a convenience wrapper around reduce_pytree(op="sum") specifically
    for differential privacy training where we need to sum clipped gradients
    from all devices before adding noise.

    Args:
        gradients: PyTree of clipped gradients to sum.

    Returns:
        gradients: PyTree with summed gradients (modified in-place).

    Example:
        >>> import torch
        >>> from opaque.distributed import sum_gradients
        >>> from opaque.clipping import clipped_grad
        >>> from opaque.noise import gaussian_noise
        >>>
        >>> # Each device computes clipped gradients on local batch
        >>> # (Poisson sampling: different batch sizes!)
        >>> loss_fn = lambda params, x, y: ((x @ params - y) ** 2).mean()
        >>> grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0, batch_argnums=(1, 2))
        >>>
        >>> # Rank 0: 32 examples, Rank 1: 28 examples
        >>> grads = grad_fn(params, batch_x, batch_y)  # Sum of B clipped grads
        >>>
        >>> # Sum across devices: total 60 examples
        >>> grads = sum_gradients(grads)
        >>>
        >>> # Add noise scaled to clip_norm (NOT dependent on batch size!)
        >>> noisy_grads = gaussian_noise(grads, sigma=1.1)

    Notes:
        - Essential for DP-SGD with Poisson sampling across devices
        - Clipped gradients have sensitivity C (independent of batch size)
        - Noise is added AFTER summing (scaled to C, not B*C)
        - If distributed is not initialized, returns input unchanged
    """
    gradients, _ = reduce_pytree(gradients, op="sum", async_op=False)
    return gradients
