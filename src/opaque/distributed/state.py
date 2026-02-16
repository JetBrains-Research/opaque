"""State synchronization for distributed DP training.

This module provides functions to synchronize scalar values and state objects
across devices. Particularly useful for adaptive clipping where clip_norm
needs to be consistent across all devices.
"""

from dataclasses import fields, is_dataclass
from typing import Any

import torch
import torch.distributed as dist

from . import all_reduce as all_reduce_tensor
from . import is_initialized

__all__ = [
    "sync_scalar",
    "sync_state",
]


def sync_scalar(
    value: float,
    op: str = "mean",
    device: torch.device | None = None,
) -> float:
    """Synchronize a scalar value across all processes.

    Args:
        value: Scalar value to synchronize.
        op: Reduction operation. One of: "sum", "mean", "max", "min", "product".
            Default: "mean" (most common for statistics like clip_norm).
        device: Device to place tensor on. If None, uses CPU.
            Default: None.

    Returns:
        float: Synchronized scalar value.

    Example:
        >>> import opaque.distributed.state as dist_state
        >>> import torch.distributed as dist
        >>>
        >>> # Initialize distributed (rank 0 of 2)
        >>> dist.init_process_group(backend='nccl', rank=0, world_size=2)
        >>>
        >>> # Each device has a different clip_norm
        >>> clip_norm_rank0 = 1.0  # Rank 0
        >>> clip_norm_rank1 = 1.2  # Rank 1
        >>>
        >>> # Synchronize (average across devices)
        >>> clip_norm = dist_state.sync_scalar(clip_norm_rank0, op="mean")
        >>> # clip_norm is now 1.1 on both devices

    Notes:
        - If distributed is not initialized, returns input unchanged
        - Typical use: sync adaptive clipping state after each step
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
                        "provide an explicit `device` to `sync_scalar` or initialize "
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
            synced_value = sync_scalar(value, op=op, device=device)
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
