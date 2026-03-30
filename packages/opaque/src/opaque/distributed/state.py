"""State synchronization and gathering for distributed DP training.

This module provides functions to reduce scalars and gather tensors
across devices. Particularly useful for adaptive clipping where per-example
norms need to be gathered from all devices.
"""

from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from typing import Any

import torch
import torch.distributed as dist

from opaque.utils.pytree import tree_map

from . import all_reduce_ as all_reduce_tensor
from . import get_world_size, is_distributed

__all__ = [
    "assert_pytree_equal",
    "assert_scalar_equal",
    "gather_pytree",
    "gather_tensors",
    "reduce_scalar",
    "sync_object",
]


def assert_scalar_equal(
    value: float | int,
    *,
    name: str,
    atol: float = 1e-8,
    rtol: float = 1e-5,
    device: torch.device | None = None,
) -> None:
    """Assert that a scalar value is equal across distributed ranks.

    Args:
        value: Local scalar value to validate.
        name: Human-readable value name used in error messages.
        atol: Absolute tolerance for equality check.
        rtol: Relative tolerance for equality check.
        device: Optional device for scalar reductions.

    Raises:
        RuntimeError: If values differ across ranks beyond tolerance.
    """
    if not is_distributed():
        return

    min_value = reduce_scalar(float(value), op="min", device=device)
    max_value = reduce_scalar(float(value), op="max", device=device)
    if not torch.isclose(
        torch.tensor(min_value),
        torch.tensor(max_value),
        atol=atol,
        rtol=rtol,
    ):
        raise RuntimeError(
            f"{name} mismatch across ranks: min={min_value}, max={max_value}."
        )


def assert_pytree_equal(
    pytree: Any,
    *,
    name: str = "pytree",
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    """Assert that a pytree of tensors is identical across all ranks.

    Uses a cheap fingerprint (sum of all elements) to detect divergence
    with a single scalar all-reduce, rather than transferring the full pytree.

    This is useful for debugging to verify that model parameters or gradients
    haven't diverged across ranks.

    Args:
        pytree: Nested dict/list of tensors to check.
        name: Human-readable name for error messages.
        atol: Absolute tolerance for fingerprint comparison.
        rtol: Relative tolerance for fingerprint comparison.

    Raises:
        RuntimeError: If the fingerprint differs across ranks.

    Example:
        >>> from opaque.distributed import assert_pytree_equal
        >>> assert_pytree_equal(model_params, name="model_params")
    """
    if not is_distributed():
        return

    # Compute a fingerprint: sum of all elements across all tensors
    total = 0.0

    def accumulate(leaf: Any) -> Any:
        nonlocal total
        if isinstance(leaf, torch.Tensor):
            total += leaf.detach().double().sum().item()
        return leaf

    tree_map(accumulate, pytree)

    assert_scalar_equal(total, name=name, atol=atol, rtol=rtol)


def gather_pytree(pytree: Any) -> Any:
    """Gather tensor leaves from all devices and concatenate along batch dimension.

    This is a best-effort helper for per-example auxiliary outputs. Leaves that
    are None are preserved. Non-tensor, non-None leaves are not supported in
    distributed mode and will raise a TypeError.

    Args:
        pytree: PyTree with tensor leaves or None leaves.

    Returns:
        PyTree with gathered tensor leaves.
    """
    if not is_distributed():
        return pytree

    def gather_leaf(leaf: Any) -> Any:
        if leaf is None:
            return None
        if isinstance(leaf, torch.Tensor):
            return gather_tensors(leaf, dim=0)
        raise TypeError(
            f"Distributed aux gathering supports tensor leaves only; got {type(leaf)}."
        )

    return tree_map(gather_leaf, pytree)


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
    if not is_distributed():
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
    all_reduce_tensor(tensor, op=op)

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
    if not is_distributed():
        return tensor

    # Gather tensors from all devices (handles variable sizes)
    gathered = [None] * get_world_size()
    dist.all_gather_object(gathered, tensor.cpu())

    # Concatenate along specified dimension
    gathered_tensors: list[torch.Tensor] = [
        t.to(tensor.device) for t in gathered if t is not None
    ]
    return torch.cat(gathered_tensors, dim=dim)


def sync_object(
    state: Any,
    field_ops: Mapping[str, str | Callable[..., float]] | None = None,
    device: torch.device | None = None,
) -> Any:
    """Synchronize scalar fields in a dataclass state object across processes.

    This is particularly useful for adaptive clipping state where we need to
    keep clipping_norm consistent across all devices.

    Args:
        state: Dataclass instance with scalar fields to synchronize.
                field_ops: Per-field reduction operations. Keys are field names.
                        Values can be either:
                        - an op string: "sum", "mean", "max", "min", "product"
                            (passed to `reduce_scalar`), or "assert_equal" to validate
                            the value is identical across ranks (raises on mismatch), or
                        - a callable for custom reduction. Callable can be either
                            `fn(value)` or `fn(value, device)` and must return a synchronized
                            scalar float value.
                        If None, all numeric fields are synchronized with "mean".
        device: Device to place tensors on. If None, uses CPU. Default: None.

    Returns:
        New state object with synchronized fields (immutable update).

    Raises:
        TypeError: If state is not a dataclass.
        ValueError: If field_ops contains non-existent field names.

    Example:
        >>> from dataclasses import dataclass
        >>> import opaque.distributed.state as dist_state
        >>>
        >>> @dataclass
        >>> class TrainingState:
        ...     next_clipping_norm: float
        ...     step: int
        ...     num_clipped: float
        >>>
        >>> state = TrainingState(next_clipping_norm=1.0, step=100, num_clipped=5.0)
        >>>
        >>> # Synchronize num_clipped (sum) and next_clipping_norm (mean), but not step
        >>> state = dist_state.sync_object(
        ...     state,
        ...     field_ops={"next_clipping_norm": "mean", "num_clipped": "sum"}
        ... )
        >>> # state.step unchanged (not synchronized)

    Notes:
        - Returns a NEW state object (immutable update)
        - If distributed is not initialized, returns input unchanged
        - Only synchronizes numeric (float/int) fields
        - Step counters typically should NOT be synchronized unless explicitly requested
    """
    if not is_distributed():
        return state

    if not is_dataclass(state):
        raise TypeError(f"state must be a dataclass, got {type(state)}")

    state_fields = {f.name for f in fields(state)}

    if field_ops is None:
        # Default: sync all numeric (float/int) fields with mean, excluding bools
        field_ops = {}
        for f in fields(state):
            val = getattr(state, f.name)
            if isinstance(val, (float, int)) and not isinstance(val, bool):
                field_ops[f.name] = "mean"
    else:
        invalid_fields = set(field_ops) - state_fields
        if invalid_fields:
            raise ValueError(
                f"field_ops contains non-existent fields: {invalid_fields}. "
                f"Available fields: {state_fields}"
            )

    # Synchronize each field
    updates = {}
    for field_name, field_op in field_ops.items():
        value = getattr(state, field_name)
        if isinstance(value, (float, int)):
            if field_op == "assert_equal":
                assert_scalar_equal(
                    float(value), name=f"{type(state).__name__}.{field_name}"
                )
                continue
            elif isinstance(field_op, str):
                synced_value = reduce_scalar(value, op=field_op, device=device)
            elif callable(field_op):
                numeric_value = float(value)
                try:
                    synced_value = field_op(numeric_value, device)
                except TypeError:
                    synced_value = field_op(numeric_value)
            else:
                raise TypeError(
                    f"field_ops[{field_name}] must be str or callable, "
                    f"got {type(field_op)}"
                )

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
