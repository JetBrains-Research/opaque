"""Gradient norm buffer for adaptive clipping threshold computation.

This module implements efficient tracking of gradient norms to compute
percentile-based adaptive clipping thresholds, as described in DP-Adam-AC.
"""

from collections import deque
from typing import Optional

import torch


class ClipNormBuffer:
    """Efficient buffer for tracking gradient norms and computing percentiles.

    This buffer maintains a rolling window of recent gradient norms (normalized
    by batch size) to compute adaptive clipping thresholds based on percentiles.

    The adaptive clipping algorithm (from DP-Adam-AC paper):
        1. Track unit-normalized gradient norms: u_i = ||g_i||_2 / max(1, |B_i|)
        2. Keep history of recent H norms in buffer
        3. Compute clip norm C as q-th percentile of buffer
        4. Adjust q based on target clip rate ρ*

    Attributes:
        capacity: Maximum number of norms to store (H in paper)
        buffer: Deque storing recent unit-normalized gradient norms
        target_clip_rate: Target fraction of gradients to clip (ρ*)

    Example:
        >>> buffer = ClipNormBuffer(capacity=1000, target_clip_rate=0.20)
        >>>
        >>> # After computing per-example gradients
        >>> pre_clip_norms = torch.tensor([0.8, 1.2, 0.9, 1.5, 0.7])
        >>> batch_sizes = torch.ones(5)  # All examples same size
        >>>
        >>> # Update buffer with new norms
        >>> buffer.update(pre_clip_norms, batch_sizes)
        >>>
        >>> # Get adaptive clip norm (80th percentile for ρ*=0.20)
        >>> clip_norm = buffer.get_adaptive_clip_norm()
        >>> print(f"Adaptive C = {clip_norm:.2f}")
        >>>
        >>> # Check actual clip rate
        >>> clip_rate = buffer.get_clip_rate(clip_norm)
        >>> print(f"Clip rate: {clip_rate:.1%}")
    """

    def __init__(
        self,
        capacity: int = 1000,
        target_clip_rate: float = 0.20,
    ):
        """Initialize gradient norm buffer.

        Args:
            capacity: Maximum number of gradient norms to store (H in paper).
                Larger values give more stable percentiles but slower adaptation.
                Default: 1000 (as used in paper)
            target_clip_rate: Target fraction of gradients to clip (ρ*).
                Controls the percentile used for adaptive threshold.
                Default: 0.20 (clip 20% of gradients)

        Raises:
            ValueError: If capacity <= 0 or target_clip_rate not in (0, 1)
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if not 0 < target_clip_rate < 1:
            raise ValueError(
                f"target_clip_rate must be in (0, 1), got {target_clip_rate}"
            )

        self.capacity = capacity
        self.target_clip_rate = target_clip_rate
        self.buffer: deque[float] = deque(maxlen=capacity)

    def update(
        self,
        pre_clip_norms: torch.Tensor,
        batch_sizes: Optional[torch.Tensor] = None,
    ) -> None:
        """Add new gradient norms to the buffer.

        Computes unit-normalized norms: u_i = ||g_i||_2 / max(1, |B_i|)
        and adds them to the rolling buffer.

        Args:
            pre_clip_norms: Tensor of gradient L2 norms before clipping.
                Shape: (num_examples,) or scalar
            batch_sizes: Tensor of batch sizes for each example (for microbatching).
                If None, assumes all examples have size 1.
                Shape: (num_examples,) or scalar

        Example:
            >>> buffer = ClipNormBuffer()
            >>>
            >>> # Single batch of 5 examples
            >>> norms = torch.tensor([0.8, 1.2, 0.9, 1.5, 0.7])
            >>> buffer.update(norms)  # Assumes batch_sizes=1 for all
            >>>
            >>> # With microbatching (variable sizes)
            >>> norms = torch.tensor([2.4, 1.8, 3.0])
            >>> sizes = torch.tensor([2.0, 2.0, 3.0])
            >>> buffer.update(norms, sizes)  # Unit norms: [1.2, 0.9, 1.0]
        """
        # Ensure tensors
        if not isinstance(pre_clip_norms, torch.Tensor):
            pre_clip_norms = torch.tensor([pre_clip_norms])
        if pre_clip_norms.dim() == 0:
            pre_clip_norms = pre_clip_norms.unsqueeze(0)

        # Default batch sizes to 1 (ensure same device as pre_clip_norms)
        if batch_sizes is None:
            batch_sizes = torch.ones_like(pre_clip_norms)
        elif not isinstance(batch_sizes, torch.Tensor):
            batch_sizes = torch.tensor([batch_sizes], device=pre_clip_norms.device, dtype=pre_clip_norms.dtype)
        if batch_sizes.dim() == 0:
            batch_sizes = batch_sizes.unsqueeze(0)

        # Compute unit-normalized gradient norms: u_i = ||g_i|| / max(1, |B_i|)
        # This normalizes by batch size to make norms comparable across different batch sizes
        denominators = torch.maximum(batch_sizes, torch.ones_like(batch_sizes))
        unit_norms = pre_clip_norms / denominators

        # Add to buffer (deque automatically handles capacity limit)
        self.buffer.extend(unit_norms.tolist())

    def get_adaptive_clip_norm(
        self,
        clip_norm_min: float = 0.1,
        clip_norm_max: float = 10.0,
    ) -> float:
        """Compute adaptive clipping threshold as percentile of buffer.

        Uses the formula from DP-Adam-AC paper:
            q = 100 * (1 - ρ*)
            C = Percentile_q(buffer)

        Args:
            clip_norm_min: Minimum allowed clip norm (C_min). Default: 0.1
            clip_norm_max: Maximum allowed clip norm (C_max). Default: 10.0

        Returns:
            Adaptive clipping threshold C, clamped to [C_min, C_max].
            Returns 1.0 if buffer is empty.

        Example:
            >>> buffer = ClipNormBuffer(target_clip_rate=0.20)
            >>> buffer.update(torch.randn(100).abs())
            >>>
            >>> # Get 80th percentile (100 * (1 - 0.20))
            >>> clip_norm = buffer.get_adaptive_clip_norm()
            >>> print(f"C = {clip_norm:.2f}")
        """
        if not self.buffer:
            return 1.0  # Default when no history

        # Compute target percentile: q = 1 - ρ*
        # Example: ρ*=0.20 → q=0.80 → 80th percentile
        quantile = 1.0 - self.target_clip_rate

        # Compute percentile of buffer
        buffer_tensor = torch.tensor(list(self.buffer))
        clip_norm = torch.quantile(buffer_tensor, quantile).item()

        # Clamp to valid range
        clip_norm = max(clip_norm_min, min(clip_norm_max, clip_norm))

        return float(clip_norm)

    def get_clip_rate(self, threshold: float) -> float:
        """Compute fraction of norms exceeding threshold.

        This is the empirical clipping rate ρ used to adjust learning rate
        in DP-Adam-AC.

        Args:
            threshold: Clipping threshold C to evaluate

        Returns:
            Fraction of norms in buffer that exceed threshold, in [0, 1].
            Returns 0.0 if buffer is empty.

        Example:
            >>> buffer = ClipNormBuffer()
            >>> buffer.update(torch.tensor([0.5, 1.2, 0.8, 1.5, 0.9]))
            >>>
            >>> # Check clip rate at C=1.0
            >>> rate = buffer.get_clip_rate(1.0)
            >>> print(f"Clip rate: {rate:.1%}")  # 40% (2 out of 5)
        """
        if not self.buffer:
            return 0.0

        # Count norms exceeding threshold
        num_clipped = sum(1 for norm in self.buffer if norm > threshold)
        clip_rate = num_clipped / len(self.buffer)

        return float(clip_rate)

    def __len__(self) -> int:
        """Return number of norms currently in buffer."""
        return len(self.buffer)

    def clear(self) -> None:
        """Clear all norms from buffer."""
        self.buffer.clear()


__all__ = ["ClipNormBuffer"]
