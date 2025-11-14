"""Functional gradient norm buffer for adaptive clipping threshold computation.

This module implements efficient tracking of gradient norms to compute
percentile-based adaptive clipping thresholds, as described in DP-Adam-AC.

All functions are pure (no side effects). State is managed externally as
immutable tuples.
"""

from typing import Optional

import torch

# Type alias for buffer state
ClipBufferState = tuple[torch.Tensor, int]  # (norms_tensor, size)


def create(
    capacity: int = 1000,
    target_clip_rate: float = 0.20,
) -> ClipBufferState:
    """Create initial clip buffer state.

    Args:
        capacity: Maximum number of gradient norms to store (H in paper).
            Larger values give more stable percentiles but slower adaptation.
            Default: 1000 (as used in paper)
        target_clip_rate: Target fraction of gradients to clip (ρ*).
            Controls the percentile used for adaptive threshold.
            Default: 0.20 (clip 20% of gradients)

    Returns:
        Initial buffer state as (norms_tensor, size) tuple:
            - norms_tensor: Zero-initialized tensor of shape (capacity,)
            - size: Number of norms added so far (0 initially)

    Raises:
        ValueError: If capacity <= 0 or target_clip_rate not in (0, 1)

    Example:
        >>> state = create(capacity=1000, target_clip_rate=0.20)
        >>> norms_tensor, size = state
        >>> print(f"Buffer capacity: {len(norms_tensor)}, current size: {size}")
        Buffer capacity: 1000, current size: 0
    """
    if capacity <= 0:
        raise ValueError(f"capacity must be positive, got {capacity}")
    if not 0 < target_clip_rate < 1:
        raise ValueError(
            f"target_clip_rate must be in (0, 1), got {target_clip_rate}"
        )

    norms_tensor = torch.zeros(capacity)
    size = 0
    return (norms_tensor, size)


def update(
    state: ClipBufferState,
    pre_clip_norms: torch.Tensor,
    batch_sizes: Optional[torch.Tensor] = None,
) -> ClipBufferState:
    """Add new gradient norms to the buffer (pure function).

    Computes unit-normalized norms: u_i = ||g_i||_2 / max(1, |B_i|)
    and adds them to the rolling buffer.

    This function is pure: it does not modify the input state, but returns
    a new state with updated norms.

    Args:
        state: Current buffer state as (norms_tensor, size) tuple
        pre_clip_norms: Tensor of gradient L2 norms before clipping.
            Shape: (num_examples,) or scalar
        batch_sizes: Tensor of batch sizes for each example (for microbatching).
            If None, assumes all examples have size 1.
            Shape: (num_examples,) or scalar

    Returns:
        New buffer state with added norms

    Example:
        >>> state = create(capacity=5, target_clip_rate=0.20)
        >>> norms = torch.tensor([0.8, 1.2, 0.9])
        >>> state = update(state, norms)
        >>> _, size = state
        >>> print(f"Added 3 norms, current size: {size}")
        Added 3 norms, current size: 3
    """
    norms_tensor, size = state

    # Ensure tensors
    if not isinstance(pre_clip_norms, torch.Tensor):
        pre_clip_norms = torch.tensor([pre_clip_norms])
    if pre_clip_norms.dim() == 0:
        pre_clip_norms = pre_clip_norms.unsqueeze(0)

    # Default batch sizes to 1 (ensure same device as pre_clip_norms)
    if batch_sizes is None:
        batch_sizes = torch.ones_like(pre_clip_norms)
    elif not isinstance(batch_sizes, torch.Tensor):
        batch_sizes = torch.tensor(
            [batch_sizes], device=pre_clip_norms.device, dtype=pre_clip_norms.dtype
        )
    if batch_sizes.dim() == 0:
        batch_sizes = batch_sizes.unsqueeze(0)

    # Compute unit-normalized gradient norms: u_i = ||g_i|| / max(1, |B_i|)
    denominators = torch.maximum(batch_sizes, torch.ones_like(batch_sizes))
    unit_norms = pre_clip_norms / denominators

    # Clone tensor for immutability
    new_norms_tensor = norms_tensor.clone()
    capacity = len(norms_tensor)

    # Add norms using ring buffer logic
    for norm in unit_norms:
        idx = size % capacity
        new_norms_tensor[idx] = norm
        size += 1

    return (new_norms_tensor, size)


def get_adaptive_clip_norm(
    state: ClipBufferState,
    target_clip_rate: float = 0.20,
    clip_norm_min: float = 0.1,
    clip_norm_max: float = 10.0,
) -> float:
    """Compute adaptive clipping threshold as percentile of buffer (pure function).

    Uses the formula from DP-Adam-AC paper:
        q = 100 * (1 - ρ*)
        C = Percentile_q(buffer)

    Args:
        state: Current buffer state as (norms_tensor, size) tuple
        target_clip_rate: Target fraction of gradients to clip (ρ*). Default: 0.20
        clip_norm_min: Minimum allowed clip norm (C_min). Default: 0.1
        clip_norm_max: Maximum allowed clip norm (C_max). Default: 10.0

    Returns:
        Adaptive clipping threshold C, clamped to [C_min, C_max].
        Returns clip_norm_max if buffer is empty.

    Example:
        >>> state = create(capacity=100, target_clip_rate=0.20)
        >>> norms = torch.randn(50).abs()
        >>> state = update(state, norms)
        >>> clip_norm = get_adaptive_clip_norm(state, target_clip_rate=0.20)
        >>> print(f"Adaptive C = {clip_norm:.2f}")
        Adaptive C = 1.23
    """
    norms_tensor, size = state

    if size == 0:
        return clip_norm_max  # Default when no history

    # Get valid norms (handle ring buffer)
    capacity = len(norms_tensor)
    if size <= capacity:
        valid_norms = norms_tensor[:size]
    else:
        valid_norms = norms_tensor  # Buffer is full

    # Compute target percentile: q = 1 - ρ*
    # Example: ρ*=0.20 → q=0.80 → 80th percentile
    quantile = 1.0 - target_clip_rate

    # Compute percentile of valid norms
    clip_norm = torch.quantile(valid_norms, quantile).item()

    # Clamp to valid range
    clip_norm = max(clip_norm_min, min(clip_norm_max, clip_norm))

    return float(clip_norm)


def get_clip_rate(state: ClipBufferState, threshold: float) -> float:
    """Compute fraction of norms exceeding threshold (pure function).

    This is the empirical clipping rate ρ used to adjust learning rate
    in DP-Adam-AC.

    Args:
        state: Current buffer state as (norms_tensor, size) tuple
        threshold: Clipping threshold C to evaluate

    Returns:
        Fraction of norms in buffer that exceed threshold, in [0, 1].
        Returns 0.0 if buffer is empty.

    Example:
        >>> state = create(capacity=100, target_clip_rate=0.20)
        >>> norms = torch.tensor([0.5, 1.2, 0.8, 1.5, 0.9])
        >>> state = update(state, norms)
        >>> rate = get_clip_rate(state, threshold=1.0)
        >>> print(f"Clip rate: {rate:.1%}")  # 40% (2 out of 5)
        Clip rate: 40.0%
    """
    norms_tensor, size = state

    if size == 0:
        return 0.0

    # Get valid norms (handle ring buffer)
    capacity = len(norms_tensor)
    if size <= capacity:
        valid_norms = norms_tensor[:size]
    else:
        valid_norms = norms_tensor  # Buffer is full

    # Count norms exceeding threshold
    num_clipped = (valid_norms > threshold).sum().item()
    clip_rate = num_clipped / len(valid_norms)

    return float(clip_rate)


def get_size(state: ClipBufferState) -> int:
    """Get total number of norms added to buffer (pure function).

    Note: This may exceed capacity if buffer has wrapped around.

    Args:
        state: Current buffer state as (norms_tensor, size) tuple

    Returns:
        Total number of norms added (may exceed capacity)

    Example:
        >>> state = create(capacity=5, target_clip_rate=0.20)
        >>> state = update(state, torch.randn(3))
        >>> print(get_size(state))
        3
    """
    _, size = state
    return size


__all__ = [
    "ClipBufferState",
    "create",
    "update",
    "get_adaptive_clip_norm",
    "get_clip_rate",
    "get_size",
]
