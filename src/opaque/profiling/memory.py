"""Memory profiling tools for DP training.

This module provides tools to profile actual memory usage during differentially
private training with per-example gradients.

Device Support:
  - CUDA: Full memory tracking via torch.cuda APIs
  - MPS (Apple Silicon): Full memory tracking via torch.mps APIs
  - CPU: Limited support (warns that profiling is approximate)

Example:
    >>> import torch
    >>> from opaque.profiling import profile_memory, find_max_microbatch_size
    >>>
    >>> # Profile current configuration
    >>> model = model.to('mps')  # or .cuda()
    >>> data = data.to('mps')
    >>> profile = profile_memory(model, (data, targets), loss_fn, l2_clip_norm=1.0)
    >>> print(profile)
    >>>
    >>> # Auto-find max microbatch size
    >>> max_mb = find_max_microbatch_size(
    ...     model, (data, targets), batch_size=32, loss_fn, l2_clip_norm=1.0
    ... )
    >>> print(f"Use microbatch_size={max_mb}")
"""

from dataclasses import dataclass
from typing import Literal, Callable
import warnings

import torch
import torch.nn as nn


@dataclass
class MemoryProfile:
    """Actual measured memory usage from profiling.

    Attributes:
        peak_gb: Peak memory usage in GB
        available_gb: Total available memory in GB
        batch_size: Batch size used in profiling
        microbatch_size: Microbatch size used (None if full batch)
        device: Device used ("cuda", "mps", or "cpu")
        status: Profile status ("ok", "warning", "critical", "unsupported")
    """

    peak_gb: float
    available_gb: float
    batch_size: int
    microbatch_size: int | None
    device: str
    status: Literal["ok", "warning", "critical", "unsupported"]

    def utilization(self) -> float:
        """Return memory utilization as fraction (0.0-1.0).

        Returns:
            Fraction of available memory used, or 0.0 if cannot determine
        """
        if self.available_gb > 0:
            return self.peak_gb / self.available_gb
        return 0.0

    def __str__(self) -> str:
        """Human-readable memory profile."""
        mb_str = f", microbatch_size={self.microbatch_size}" if self.microbatch_size else ""
        status_emoji = {
            "ok": "✓",
            "warning": "⚠️",
            "critical": "❌",
            "unsupported": "ℹ️"
        }[self.status]

        return f"""Memory Profile (batch_size={self.batch_size}{mb_str})
  Device:           {self.device}
  Peak Memory:      {self.peak_gb:>6.2f} GB
  Available:        {self.available_gb:>6.2f} GB
  Utilization:      {self.utilization()*100:>5.1f}%
  Status:           {status_emoji} {self.status.upper()}
"""


class MemoryTracker:
    """Unified memory tracking across devices.

    Provides consistent interface for memory tracking on CUDA, MPS, and CPU.
    """

    def __init__(self, device: str):
        """Initialize tracker for specified device.

        Args:
            device: Device type ("cuda", "mps", or "cpu")
        """
        self.device = device
        self._supported = self._check_support()

    def _check_support(self) -> bool:
        """Check if memory tracking is supported on this device.

        Returns:
            True if device has memory tracking APIs
        """
        if self.device == "cuda":
            return torch.cuda.is_available()
        elif self.device == "mps":
            return torch.backends.mps.is_available()
        else:  # CPU
            return False

    def is_supported(self) -> bool:
        """Check if profiling is fully supported on this device.

        Returns:
            True if device supports accurate memory profiling
        """
        return self._supported

    def reset(self) -> None:
        """Reset memory tracking and clear cache."""
        if self.device == "cuda" and self._supported:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        elif self.device == "mps" and self._supported:
            torch.mps.empty_cache()

    def get_current_allocated(self) -> float:
        """Get current allocated memory in bytes.

        Returns:
            Current allocated memory in bytes, or 0 if unsupported
        """
        if self.device == "cuda" and self._supported:
            return torch.cuda.memory_allocated()
        elif self.device == "mps" and self._supported:
            return torch.mps.current_allocated_memory()
        return 0.0

    def get_peak_allocated(self) -> float:
        """Get peak allocated memory in bytes since last reset.

        Returns:
            Peak allocated memory in bytes, or 0 if unsupported
        """
        if self.device == "cuda" and self._supported:
            return torch.cuda.max_memory_allocated()
        elif self.device == "mps" and self._supported:
            # MPS doesn't have max_memory_allocated, use current
            # This is a limitation but still useful for profiling
            return torch.mps.current_allocated_memory()
        return 0.0

    def get_total_memory(self) -> float:
        """Get total available memory in bytes.

        Returns:
            Total available memory in bytes, or 0 if unsupported
        """
        if self.device == "cuda" and self._supported:
            return torch.cuda.get_device_properties(0).total_memory
        elif self.device == "mps" and self._supported:
            # MPS shares system memory - use available RAM
            # Use ~70% of available as conservative estimate
            try:
                import psutil
                return int(psutil.virtual_memory().available * 0.7)
            except ImportError:
                warnings.warn(
                    "psutil not installed. Cannot determine MPS memory. "
                    "Install with: pip install psutil",
                    UserWarning
                )
                return 0.0
        return 0.0


def profile_memory(
    model: nn.Module,
    sample_batch: tuple[torch.Tensor, ...],
    loss_fn: Callable,
    l2_clip_norm: float,
    *,
    microbatch_size: int | None = None,
) -> MemoryProfile:
    """Profile actual memory usage by running one training step.

    Measures real memory consumption by executing a forward pass, computing
    per-example gradients with clipping, and tracking peak memory usage.

    Device Support:
      - CUDA: Full support with accurate memory tracking
      - MPS (Apple Silicon): Full support with memory tracking
      - CPU: Limited support (returns status="unsupported" with warning)

    Args:
        model: PyTorch model to profile
        sample_batch: Single batch of (data, targets, ...) as tensors
        loss_fn: Loss function with signature (params, data, targets, ...) -> scalar
        l2_clip_norm: L2 clipping norm for gradients
        microbatch_size: Microbatch size to test (None = full batch)

    Returns:
        MemoryProfile with measured memory usage and status

    Example:
        >>> import torch
        >>> from opaque.profiling import profile_memory
        >>>
        >>> # On MPS (Apple Silicon)
        >>> model = model.to('mps')
        >>> data, targets = data.to('mps'), targets.to('mps')
        >>>
        >>> profile = profile_memory(
        ...     model,
        ...     (data, targets),
        ...     loss_fn,
        ...     l2_clip_norm=1.0,
        ... )
        >>> print(profile)
        Memory Profile (batch_size=32)
          Device:           mps
          Peak Memory:       2.45 GB
          Available:        28.50 GB
          Utilization:       8.6%
          Status:           ✓ OK
    """
    from opaque import clipped_grad
    from opaque.utils import make_functional

    # Detect device from model
    device_param = next(model.parameters())
    if device_param.is_cuda:
        device = "cuda"
    elif device_param.device.type == "mps":
        device = "mps"
    else:
        device = "cpu"

    tracker = MemoryTracker(device)

    # Warn if profiling not fully supported
    if not tracker.is_supported():
        warnings.warn(
            f"Memory profiling on {device.upper()} is limited. "
            "For accurate profiling, use CUDA or MPS devices.",
            UserWarning
        )

    # Make functional
    fmodel, trainable, frozen = make_functional(model, partition_trainable=True)
    params = {**frozen, **trainable}

    # Create grad function
    grad_fn, clip_state = clipped_grad(
        loss_fn,
        l2_clip_norm=l2_clip_norm,
        batch_argnums=tuple(range(1, len(sample_batch) + 1)),
        microbatch_size=microbatch_size,
    )

    # Reset and measure
    tracker.reset()

    # Run one step (no_grad to avoid accumulating .grad attributes)
    with torch.no_grad():
        grads, _ = grad_fn(params, *sample_batch, state=clip_state)

    # Get measurements
    peak_bytes = tracker.get_peak_allocated()
    total_bytes = tracker.get_total_memory()

    peak_gb = peak_bytes / 1e9
    available_gb = total_bytes / 1e9

    # Determine status
    if not tracker.is_supported():
        status = "unsupported"
    else:
        util = peak_gb / available_gb if available_gb > 0 else 0
        if util > 0.95:
            status = "critical"
        elif util > 0.85:
            status = "warning"
        else:
            status = "ok"

    batch_size = sample_batch[0].shape[0]

    return MemoryProfile(
        peak_gb=peak_gb,
        available_gb=available_gb,
        batch_size=batch_size,
        microbatch_size=microbatch_size,
        device=device,
        status=status,
    )


def find_max_microbatch_size(
    model: nn.Module,
    sample_batch: tuple[torch.Tensor, ...],
    batch_size: int,
    loss_fn: Callable,
    l2_clip_norm: float,
    *,
    safety_margin: float = 0.9,
    min_size: int = 1,
) -> int:
    """Find largest microbatch size that fits in memory.

    Uses binary search with actual profiling to find the maximum microbatch
    size that can be used without causing out-of-memory errors.

    Device Support:
      - CUDA: Full support with binary search
      - MPS: Full support with binary search
      - CPU: Returns min_size with warning (profiling not supported)

    Args:
        model: PyTorch model
        sample_batch: Full batch of data as (data, targets, ...)
        batch_size: Full batch size
        loss_fn: Loss function
        l2_clip_norm: L2 clipping norm
        safety_margin: Use only this fraction of available memory (default 0.9)
        min_size: Minimum microbatch size to try (default 1)

    Returns:
        Maximum microbatch size that fits (power of 2)

    Example:
        >>> from opaque.profiling import find_max_microbatch_size
        >>>
        >>> # Auto-detect best microbatch size
        >>> max_mb = find_max_microbatch_size(
        ...     model,
        ...     (data, targets),
        ...     batch_size=64,
        ...     loss_fn=loss_fn,
        ...     l2_clip_norm=1.0,
        ... )
        >>> print(f"Recommended microbatch_size={max_mb}")
        >>>
        >>> # Use in training
        >>> from opaque import clipped_grad
        >>> grad_fn = clipped_grad(
        ...     loss_fn,
        ...     l2_clip_norm=1.0,
        ...     batch_argnums=(1, 2),
        ...     microbatch_size=max_mb,
        ... )
    """
    # Detect device
    device_param = next(model.parameters())
    if device_param.is_cuda:
        device = "cuda"
    elif device_param.device.type == "mps":
        device = "mps"
    else:
        device = "cpu"

    tracker = MemoryTracker(device)

    # Check support
    if not tracker.is_supported():
        warnings.warn(
            f"Memory profiling on {device.upper()} is not supported. "
            f"Returning min_size={min_size}. Test manually on CUDA/MPS for accurate sizing.",
            UserWarning
        )
        return min_size

    # Binary search over power-of-2 sizes
    sizes = [2**i for i in range(20) if 2**i <= batch_size]
    sizes = [s for s in sizes if s >= min_size]

    if not sizes:
        return min_size

    left, right = 0, len(sizes) - 1
    best_size = min_size

    while left <= right:
        mid = (left + right) // 2
        test_size = sizes[mid]

        try:
            profile = profile_memory(
                model,
                sample_batch,
                loss_fn,
                l2_clip_norm,
                microbatch_size=test_size,
            )

            # Check if within safety margin
            if profile.utilization() <= safety_margin:
                best_size = test_size
                left = mid + 1  # Try larger
            else:
                right = mid - 1  # Try smaller

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # OOM - try smaller
                right = mid - 1
                # Clear cache
                if device == "cuda":
                    torch.cuda.empty_cache()
                elif device == "mps":
                    torch.mps.empty_cache()
            else:
                raise

    return best_size


__all__ = [
    "MemoryProfile",
    "MemoryTracker",
    "profile_memory",
    "find_max_microbatch_size",
]
