"""State synchronization and gathering for distributed DP training.

This module provides functions to reduce scalars and gather tensors
across devices. Particularly useful for adaptive clipping where per-example
norms need to be gathered from all devices.
"""

from dataclasses import fields, is_dataclass
from typing import Any

import torch
import torch.distributed as dist

from . import all_reduce as all_reduce_tensor
from . import get_world_size, is_initialized

__all__ = [
    "reduce_scalar",
    "gather_tensors",
    "sync_scalar",  # Deprecated alias
    "sync_state",  # Deprecated - will be removed
]


def reduce_scalar(
    value: float,
    op: str = "mean",
    device: torch.device | None = None,
) -> float:
    """Reduce a scalar value across all processes.

    Args:
        value: Scalar value to reduce.
        op: Reduction operation. One of: "sum", "mean", "max", "min", "product".
            Default: "mean".
        device: Device to place tensor on. If None, auto-selects based on backend.
            Default: None.

    Returns:
        float: Reduced scalar value.

    Example:
        >>> from opaque.distributed import reduce_scalar
        >>> import torch.distributed as dist
        >>>
        >>> # Initialize distributed (rank 0 of 2)
        >>> dist.init_process_group(backend='nccl', rank=0, world_size=2)
        >>>
        >>> # Each device has a different batch size
        >>> batch_size_rank0 = 32  # Rank 0
        >>> batch_size_rank1 = 28  # Rank 1 (Poisson sampling!)
        >>>
        >>> # Get total batch size across devices
        >>> total_batch_size = reduce_scalar(batch_size_rank0, op="sum")
        >>> # total_batch_size is now 60 on both devices

    Notes:
        - If distributed is not initialized, returns input unchanged
        - Typical use: sum batch sizes, average metrics
        - Creates a temporary tensor for communication
    """
    if not is_initialized():
        return value

    # Convert to tensor
    # For NCCL backend we must use CUDA tensors; prefer the current CUDA device
    # when `device` is not provided. If CUDA is unavailable while using NCCL,
    # raise an informative error rather than silently creating a CPU tensor
    # which would make the collective fail at runtime.
    if device is None:
        if dist.is_available() and dist.is_initialized():
            backend = dist.get_backend()
            if backend == "nccl":
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "Distributed backend is 'nccl' but CUDA is not available; "
                        "provide an explicit `device` to `reduce_scalar` or initialize "
                        "with a CUDA-capable process."
                    )
                # Use the currently selected CUDA device for this process.
                device = torch.device(f"cuda:{torch.cuda.current_device()}")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")

    tensor = torch.tensor(value, dtype=torch.float32, device=device)

    # All-reduce
    all_reduce_tensor(tensor, op=op, async_op=False)

    # Convert back to float
    return tensor.item()


def gather_tensors(
    tensor: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    """Gather tensors from all devices and concatenate along dimension.

    This function handles variable-size tensors across devices, which is
    essential for Poisson sampling where batch sizes differ. Uses
    `all_gather_object` internally to support arbitrary tensor sizes.

    Args:
        tensor: Local tensor to gather (any shape).
        dim: Dimension to concatenate along. Default: 0 (batch dimension).

    Returns:
        torch.Tensor: Concatenated tensor from all devices.

    Example:
        >>> from opaque.distributed import gather_tensors
        >>> import torch.distributed as dist
        >>>
        >>> # Initialize distributed (2 devices)
        >>> dist.init_process_group(backend='nccl')
        >>>
        >>> # Each device has different number of per-example norms (Poisson sampling!)
        >>> norms_rank0 = torch.tensor([1.2, 2.1, 1.8])  # 3 examples
        >>> norms_rank1 = torch.tensor([2.3, 1.9])       # 2 examples
        >>>
        >>> # Gather all norms across devices
        >>> all_norms = gather_tensors(norms_rank0, dim=0)  # On rank 0
        >>> # all_norms = tensor([1.2, 2.1, 1.8, 2.3, 1.9])  # 5 examples total
        >>>
        >>> # Compute global quantile for adaptive clipping
        >>> quantile = torch.quantile(all_norms, 0.5)

    Notes:
        - If distributed is not initialized, returns input tensor unchanged
        - Handles variable-size tensors naturally (no padding required)
        - Essential for adaptive clipping with Poisson sampling
        - Uses CPU communication via all_gather_object (moves tensors temporarily)
    """
    if not is_initialized():
        return tensor

    # Gather tensors from all devices (handles variable sizes)
    gathered = [None] * get_world_size()
    dist.all_gather_object(gathered, tensor.cpu())

    # Concatenate along specified dimension
    gathered_tensors = [t.to(tensor.device) for t in gathered]
    return torch.cat(gathered_tensors, dim=dim)


# Deprecated alias for backward compatibility
def sync_scalar(
    value: float,
    op: str = "mean",
    device: torch.device | None = None,
) -> float:
    """Deprecated: Use reduce_scalar instead.

    .. deprecated:: 2.0.0
        Use :func:`reduce_scalar` instead. This will be removed in v3.0.0.
    """
    import warnings

    warnings.warn(
        "sync_scalar is deprecated, use reduce_scalar instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return reduce_scalar(value, op=op, device=device)


def sync_state(
    state: Any,
    sync_fields: list[str] | None = None,
    op: str = "mean",
    device: torch.device | None = None,
) -> Any:
    """Synchronize scalar fields in a dataclass state object across processes.

    This is particularly useful for adaptive clipping state where we need to
    keep clip_norm consistent across all devices.

    Args:
        state: Dataclass instance with scalar fields to synchronize.
        sync_fields: List of field names to synchronize. If None, syncs all
            float/int fields. Default: None.
        op: Reduction operation for synchronization. Default: "mean".
        device: Device to place tensors on. If None, uses CPU. Default: None.

    Returns:
        New state object with synchronized fields (immutable update).

    Raises:
        TypeError: If state is not a dataclass.
        ValueError: If sync_fields contains non-existent field names.

    Example:
        >>> from dataclasses import dataclass
        >>> import opaque.distributed.state as dist_state
        >>>
        >>> @dataclass
        >>> class AdaptiveClipState:
        ...     clip_norm: float
        ...     step: int
        ...     clipping_rate: float
        >>>
        >>> # Each device has different state after local update
        >>> state = AdaptiveClipState(clip_norm=1.0, step=100, clipping_rate=0.8)
        >>>
        >>> # Synchronize clip_norm and clipping_rate (but not step)
        >>> state = dist_state.sync_state(
        ...     state,
        ...     sync_fields=["clip_norm", "clipping_rate"],
        ...     op="mean"
        ... )
        >>> # state.clip_norm and state.clipping_rate now averaged across devices
        >>> # state.step unchanged (not synchronized)

    Notes:
        - Returns a NEW state object (immutable update)
        - If distributed is not initialized, returns input unchanged
        - Only synchronizes numeric (float/int) fields
        - Step counters typically should NOT be synchronized (use sync_fields)
    """
    if not is_initialized():
        return state

    if not is_dataclass(state):
        raise TypeError(f"state must be a dataclass, got {type(state)}")

    # Determine which fields to sync
    state_fields = {f.name for f in fields(state)}

    if sync_fields is None:
        # Sync all numeric (float/int) fields by default, but exclude bools
        sync_fields = []
        for f in fields(state):
            val = getattr(state, f.name)
            if isinstance(val, (float, int)) and not isinstance(val, bool):
                sync_fields.append(f.name)
    else:
        # Validate sync_fields
        invalid_fields = set(sync_fields) - state_fields
        if invalid_fields:
            raise ValueError(
                f"sync_fields contains non-existent fields: {invalid_fields}. "
                f"Available fields: {state_fields}"
            )

    # Synchronize each field
    updates = {}
    for field_name in sync_fields:
        value = getattr(state, field_name)
        if isinstance(value, (float, int)):
            synced_value = reduce_scalar(value, op=op, device=device)
            # Preserve type (int stays int, float stays float)
            if isinstance(value, int):
                synced_value = int(synced_value)
            updates[field_name] = synced_value

    # Create new state with updated fields (immutable)
    if updates:
        # Use dataclass replace pattern
        state = type(state)(
            **{**{f.name: getattr(state, f.name) for f in fields(state)}, **updates}
        )

    return state
