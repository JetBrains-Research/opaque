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
    >>> grad_fn = clipped_grad(loss_fn, clipping_norm=1.0, batch_argnums=(1, 2))
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
    ...     initial_clipping_norm=1.0,
    ... )
    >>>
    >>> # In distributed mode, gathers ALL per-example norms from ALL devices
    >>> # to compute global quantile (clipping_norm identical everywhere)
    >>> grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
"""

import torch
import torch.distributed as dist

__all__ = [
    # Core utilities
    "is_distributed",
    "get_rank",
    "get_world_size",
    "all_reduce",
    "all_reduce_",
    "barrier",
    # PyTree reduction
    "reduce_pytree",
    "reduce_pytree_",
    # DP-specific helpers
    "sum_gradients",
    "sum_gradients_",
    # Scalar reduction
    "reduce_scalar",
    "assert_pytree_equal",
    "assert_scalar_equal",
    # Tensor gathering
    "gather_tensors",
    "gather_pytree",
    # State synchronization
    "sync_object",
    "sync",
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


def all_reduce_(
    tensor: torch.Tensor,
    op: str = "sum",
) -> None:
    """All-reduce a tensor across all processes (in-place).

    This is a thin blocking wrapper over ``torch.distributed.all_reduce``.
    The input tensor is mutated in-place and no value is returned.

    Args:
        tensor: Tensor to reduce. Modified in-place.
        op: Reduction operation. One of: "sum", "mean", "max", "min", "product".
            Default: "sum".
    Returns:
        None. The operation is always blocking.

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
        >>> dist_utils.all_reduce_(t, op="sum")
        >>> print(t)  # [2.0, 4.0, 6.0] (sum of rank 0 and rank 1)
        >>>
        >>> # Average across all devices
        >>> t = torch.tensor([1.0, 2.0, 3.0])
        >>> dist_utils.all_reduce_(t, op="mean")
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

    dist.all_reduce(tensor, op=op_map[op])


def all_reduce(
    tensor: torch.Tensor,
    op: str = "sum",
) -> torch.Tensor:
    """All-reduce a tensor across all processes and return a reduced copy.

    This is the functional counterpart to ``all_reduce_``.

    Args:
        tensor: Tensor to reduce.
        op: Reduction operation. One of: "sum", "mean", "max", "min", "product".
            Default: "sum".

    Returns:
        Reduced tensor copy. The input tensor is unchanged.
    """
    reduced = tensor.clone()
    all_reduce_(reduced, op=op)
    return reduced


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
    reduce_pytree_,
    sum_gradients,
    sum_gradients_,
)
from .state import (  # noqa: E402
    assert_pytree_equal,
    assert_scalar_equal,
    gather_pytree,
    gather_tensors,
    reduce_scalar,
    sync_object,
)

# ---- Type-based sync dispatcher ----
# Maps type → sync function. Populated by clipping and noise modules
# when they are imported (via register_sync_type).
_SYNC_REGISTRY: dict[type, object] = {}


def register_sync_type(state_type: type, sync_fn: object) -> None:
    """Register a sync function for a given state/aux type."""
    _SYNC_REGISTRY[state_type] = sync_fn


def sync(*states: object) -> object | tuple[object, ...]:
    """Synchronize one or more state/auxiliary objects across distributed ranks.

    Auto-dispatches to the right specialized sync function based on each
    object's type. Works with clipping states/aux, noise states, and
    profiling objects.

    Args:
        *states: One or more registered state/aux objects.

    Returns:
        - If one argument is provided: synchronized object.
        - If multiple arguments are provided: tuple of synchronized objects
          in the same order.

    Raises:
        TypeError: If no sync function is registered for the type.

    Example::

        from opaque.distributed import sync

        clip_state = sync(clip_state)       # dispatches to sync_clip_state
        noise_state = sync(noise_state)     # dispatches to sync_gaussian_noise_state
        aux = sync(aux)                     # dispatches to aux sync handler
        clip_state, aux = sync(clip_state, aux)
    """

    def _sync_one(single_state: object) -> object:
        state_type = type(single_state)
        if state_type not in _SYNC_REGISTRY:
            import opaque.clipping.distributed  # noqa: F401
            import opaque.noise.distributed  # noqa: F401
            import opaque.profiling.distributed  # noqa: F401
        if state_type in _SYNC_REGISTRY:
            return _SYNC_REGISTRY[state_type](single_state)
        raise TypeError(
            f"No sync function registered for {state_type.__name__}. "
            f"Registered types: {[t.__name__ for t in _SYNC_REGISTRY]}"
        )

    if len(states) == 1:
        return _sync_one(states[0])
    if len(states) > 1:
        return tuple(_sync_one(single_state) for single_state in states)
    return ()
