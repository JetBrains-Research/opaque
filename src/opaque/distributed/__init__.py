"""Distributed training utilities for differential privacy.

This module provides PyTorch-native distributed primitives for DP training:
- Core utilities: check initialization, get rank/world_size
- PyTree reduction: reduce_pytree for generic aggregation
- Gradient helpers: sum_gradients for DP-specific gradient summing
- Scalar reduction: reduce_scalar for batch sizes, metrics
- Tensor gathering: gather_tensors for adaptive clipping with variable batch sizes

Design Philosophy:
    - Composable primitives, not heavyweight abstractions
    - PyTorch-native patterns (DDP)
    - No custom backward hooks (functional API already produces clipped gradients)
    - Automatic distributed detection (no manual configuration)

Example - Standard DP-SGD with Poisson Sampling:
    >>> import opaque.distributed as dist_utils
    >>> import torch.distributed as dist
    >>> from opaque.clipping import clipped_grad
    >>> from opaque.noise import gaussian_noise
    >>>
    >>> # Initialize distributed
    >>> dist.init_process_group(backend='nccl')
    >>>
    >>> # Each device: compute clipped gradients on local batch (Poisson sampling)
    >>> grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0, batch_argnums=(1, 2))
    >>> grads = grad_fn(params, batch_x, batch_y)  # Sum of B_local clipped grads
    >>>
    >>> # Sum across devices: total gradients from all examples
    >>> grads = dist_utils.sum_gradients(grads)
    >>>
    >>> # Add noise (sensitivity = C, NOT batch-dependent!)
    >>> noise_fn, noise_state = gaussian_noise(stddev=1.1)
    >>> noisy_grads, noise_state = noise_fn(grads, noise_state)

Example - Adaptive Clipping (Automatic Distributed Detection):
    >>> from opaque.clipping import adaptive_clipped_grad
    >>>
    >>> # Adaptive clipping automatically detects distributed mode!
    >>> grad_fn, clip_state = adaptive_clipped_grad(
    ...     loss_fn,
    ...     batch_argnums=(1, 2),
    ...     initial_clip_norm=1.0,
    ... )
    >>>
    >>> # In distributed mode, gathers ALL per-example norms from ALL devices
    >>> # to compute global quantile (clip_norm identical everywhere)
    >>> grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
"""

from typing import Optional

import torch
import torch.distributed as dist

__all__ = [
    # Core utilities
    "is_distributed",
    "get_rank",
    "get_world_size",
    "all_reduce",
    "barrier",
    # PyTree reduction
    "reduce_pytree",
    # DP-specific helpers
    "sum_gradients",
    # Scalar reduction
    "reduce_scalar",
    "assert_scalar_equal",
    # Tensor gathering
    "gather_tensors",
    "gather_pytree",
    "gather_pytree_tensors",
    # State synchronization
    "sync_state",
]


def is_distributed() -> bool:
    """Check if PyTorch distributed training is initialized.

    Returns:
        bool: True if torch.distributed.is_initialized(), False otherwise.

    Example:
        >>> import torch.distributed as dist
        >>> import opaque.distributed as dist_utils
        >>>
        >>> # Before init
        >>> dist_utils.is_distributed()
        False
        >>>
        >>> # After init
        >>> dist.init_process_group(backend='nccl')
        >>> dist_utils.is_distributed()
        True
    """
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get current process rank in distributed training.

    Returns:
        int: Rank (0 to world_size-1) if distributed, else 0.

    Example:
        >>> import opaque.distributed as dist_utils
        >>>
        >>> rank = dist_utils.get_rank()
        >>> print(f"Process rank: {rank}")
        Process rank: 0
    """
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes in distributed training.

    Returns:
        int: World size (number of devices) if distributed, else 1.

    Example:
        >>> import opaque.distributed as dist_utils
        >>>
        >>> world_size = dist_utils.get_world_size()
        >>> print(f"Training on {world_size} devices")
        Training on 1 devices
    """
    if is_distributed():
        return dist.get_world_size()
    return 1


def all_reduce(
    tensor: torch.Tensor,
    op: str = "sum",
    async_op: bool = False,
) -> dist.Work | None:
    """All-reduce a tensor across all processes (in-place).

    Args:
        tensor: Tensor to reduce. Modified in-place.
        op: Reduction operation. One of: "sum", "mean", "max", "min", "product".
            Default: "sum".
        async_op: If True, return a work handle for asynchronous operation.
            Default: False (blocking).

    Returns:
        Optional[dist.Work]: Work handle if async_op=True, else None.

    Raises:
        ValueError: If op is not a valid reduction operation.
        RuntimeError: If distributed is not initialized.

    Example:
        >>> import torch
        >>> import torch.distributed as dist
        >>> import opaque.distributed as dist_utils
        >>>
        >>> # Initialize distributed (rank 0 of 2)
        >>> dist.init_process_group(backend='nccl', rank=0, world_size=2)
        >>>
        >>> # Each device has a different tensor
        >>> t = torch.tensor([1.0, 2.0, 3.0])
        >>>
        >>> # Sum across all devices (in-place)
        >>> dist_utils.all_reduce(t, op="sum")
        >>> print(t)  # [2.0, 4.0, 6.0] (sum of rank 0 and rank 1)
        >>>
        >>> # Average across all devices
        >>> t = torch.tensor([1.0, 2.0, 3.0])
        >>> dist_utils.all_reduce(t, op="mean")
        >>> print(t)  # [1.0, 2.0, 3.0] (average of rank 0 and rank 1)
    """
    # Map string to ReduceOp
    op_map = {
        "sum": dist.ReduceOp.SUM,
        "mean": dist.ReduceOp.AVG,
        "max": dist.ReduceOp.MAX,
        "min": dist.ReduceOp.MIN,
        "product": dist.ReduceOp.PRODUCT,
    }

    if op not in op_map:
        raise ValueError(
            f"Invalid reduction operation: {op}. Must be one of: {list(op_map.keys())}"
        )

    if not is_distributed():
        raise RuntimeError(
            "torch.distributed is not initialized. "
            "Call torch.distributed.init_process_group() first."
        )

    return dist.all_reduce(tensor, op=op_map[op], async_op=async_op)


def barrier() -> None:
    """Synchronize all processes (blocking barrier).

    All processes wait until all processes reach this call.

    Example:
        >>> import opaque.distributed as dist_utils
        >>>
        >>> # Process 0: load model (slow)
        >>> if dist_utils.get_rank() == 0:
        ...     model = load_large_model()
        >>>
        >>> # Wait for rank 0 to finish loading
        >>> dist_utils.barrier()
        >>>
        >>> # All processes continue together
        >>> train(model)
    """
    if is_distributed():
        dist.barrier()


# Import submodules AFTER core functions are defined (avoid circular import)
from .gradients import (  # noqa: E402
    reduce_pytree,
    sum_gradients,
)
from .state import (  # noqa: E402
    assert_scalar_equal,
    gather_pytree,
    gather_pytree_tensors,
    gather_tensors,
    reduce_scalar,
    sync_state,
)
