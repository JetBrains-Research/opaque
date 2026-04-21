"""Memory and timing profiling tools for DP training.

This module provides lightweight tools for tracking memory usage and timing
during differentially private training. Profiling history is modeled as
explicit immutable state so it can be threaded through training loops and
safely synchronized across distributed ranks.

Key Components:
    - MemoryStats: Dataclass for GPU memory statistics
    - StepTimer: Context manager for timing individual training steps
    - TrainingProfiler: Immutable profiler state for memory + timing history
    - Utility functions: get_memory_stats, print_memory, reset_peak_memory

Example - Basic usage in training loop:
    >>> from opaque.performance.profiling import StepTimer, TrainingProfiler
    >>>
    >>> profiler = TrainingProfiler(device)
    >>> profiler, _ = profiler.mark("model_loaded")
    >>>
    >>> for step, batch in enumerate(dataloader):
    ...     timer = StepTimer(device, batch_size=len(batch))
    ...     with timer:
    ...         grads = compute_gradients(batch)
    ...         update_params(grads)
    ...     profiler = profiler.add_step(timer)
    ...
    ...     if step % 10 == 0:
    ...         print(profiler.step_summary())
    ...         wandb.log(profiler.current_metrics())

Example - Simple memory tracking:
    >>> from opaque.performance.profiling import get_memory_stats, print_memory
    >>>
    >>> print_memory(device, "After model load")
    >>> # ... do work ...
    >>> stats = get_memory_stats(device)
    >>> print(f"Peak: {stats.peak_gb:.2f} GB")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import torch


@dataclass(frozen=True)
class MemoryStats:
    """GPU memory statistics at a point in time.

    All values are in GB for easy reading. Use to_dict() for WANDB logging.

    Attributes:
        allocated_gb: Currently allocated memory
        reserved_gb: Reserved by PyTorch allocator (includes fragmentation)
        peak_gb: Peak allocated since last reset
        free_gb: Estimated free memory (total - reserved)
        total_gb: Total GPU memory
    """

    allocated_gb: float = 0.0
    reserved_gb: float = 0.0
    peak_gb: float = 0.0
    free_gb: float = 0.0
    total_gb: float = 0.0
    exact_peak: bool = False
    exact_reserved: bool = False
    known_free: bool = False
    known_total: bool = False

    @property
    def utilization(self) -> float:
        """Memory utilization as fraction (0.0-1.0)."""
        if self.total_gb > 0:
            return self.peak_gb / self.total_gb
        return 0.0

    def to_dict(self, prefix: str = "memory/") -> dict[str, float | bool]:
        """Convert to dict for WANDB logging.

        Args:
            prefix: Prefix for all keys (default: "memory/")

        Returns:
            Dict with keys like "memory/allocated_gb", "memory/peak_gb", etc.
        """
        return {
            f"{prefix}allocated_gb": self.allocated_gb,
            f"{prefix}reserved_gb": self.reserved_gb,
            f"{prefix}peak_gb": self.peak_gb,
            f"{prefix}free_gb": self.free_gb,
            f"{prefix}total_gb": self.total_gb,
            f"{prefix}utilization": self.utilization,
            f"{prefix}peak_exact": self.exact_peak,
            f"{prefix}reserved_exact": self.exact_reserved,
            f"{prefix}free_known": self.known_free,
            f"{prefix}total_known": self.known_total,
        }

    def __str__(self) -> str:
        return (
            f"Memory: alloc={self.allocated_gb:.2f}GB, "
            f"peak={self.peak_gb:.2f}GB, "
            f"free={self.free_gb:.2f}GB "
            f"({self.utilization:.1%} used)"
        )


def get_memory_stats(device: torch.device | str) -> MemoryStats:
    """Get current GPU memory statistics.

    Args:
        device: PyTorch device (cuda, mps, or cpu)

    Returns:
        MemoryStats with current memory usage. All zeros for CPU.

    Example:
        >>> stats = get_memory_stats("cuda")
        >>> print(f"Peak memory: {stats.peak_gb:.2f} GB")
    """
    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        peak = torch.cuda.max_memory_allocated(device) / (1024**3)
        total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        free = total - reserved
        return MemoryStats(
            allocated_gb=allocated,
            reserved_gb=reserved,
            peak_gb=peak,
            free_gb=free,
            total_gb=total,
            exact_peak=True,
            exact_reserved=True,
            known_free=True,
            known_total=True,
        )
    elif device.type == "mps":
        allocated = torch.mps.current_allocated_memory() / (1024**3)
        # MPS doesn't have peak tracking, use current as approximation
        return MemoryStats(
            allocated_gb=allocated,
            reserved_gb=allocated,
            peak_gb=allocated,
            free_gb=0.0,  # MPS doesn't expose this
            total_gb=0.0,
            exact_peak=False,
            exact_reserved=False,
            known_free=False,
            known_total=False,
        )
    else:
        return MemoryStats(
            exact_peak=False,
            exact_reserved=False,
            known_free=False,
            known_total=False,
        )


def reset_peak_memory(device: torch.device | str) -> None:
    """Reset peak memory counter for accurate per-phase tracking.

    Args:
        device: PyTorch device

    Example:
        >>> reset_peak_memory("cuda")
        >>> # ... do training step ...
        >>> stats = get_memory_stats("cuda")
        >>> print(f"Step peak: {stats.peak_gb:.2f} GB")
    """
    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # MPS doesn't support peak memory reset


def print_memory(device: torch.device | str, label: str = "") -> MemoryStats:
    """Print memory stats with optional label. Returns stats for further use.

    Args:
        device: PyTorch device
        label: Optional label to prefix the output

    Returns:
        MemoryStats for the current state

    Example:
        >>> print_memory("cuda", "After model load")
        [After model load] Memory: alloc=7.50GB, peak=7.50GB, free=72.50GB (9.4% used)
    """
    stats = get_memory_stats(device)
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}{stats}")
    return stats


def empty_cache(device: torch.device | str) -> None:
    """Clear GPU cache to free reserved memory.

    Args:
        device: PyTorch device
    """
    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


@dataclass(frozen=True)
class StepMetrics:
    """Metrics for a single training step.

    Attributes:
        step_time: Wall-clock time for the step in seconds
        peak_memory_gb: Peak GPU memory during the step
        batch_size: Number of samples processed
        throughput: Samples per second
    """

    step_time: float = 0.0
    peak_memory_gb: float = 0.0
    batch_size: int = 0
    throughput: float = 0.0

    def to_dict(self, prefix: str = "perf/") -> dict[str, float]:
        """Convert to dict for WANDB logging."""
        return {
            f"{prefix}step_time_sec": self.step_time,
            f"{prefix}peak_memory_gb": self.peak_memory_gb,
            f"{prefix}throughput_samples_sec": self.throughput,
        }


class StepTimer:
    """Context manager for timing a training step with GPU sync.

    Handles GPU synchronization for accurate timing and optional
    peak memory tracking.

    Example:
        >>> timer = StepTimer("cuda")
        >>> with timer:
        ...     grads = compute_gradients(batch)
        ...     update_params(grads)
        >>> print(f"Step took {timer.elapsed:.2f}s")
    """

    def __init__(
        self,
        device: torch.device | str,
        *,
        track_memory: bool = True,
        batch_size: int = 0,
    ):
        """Initialize step timer.

        Args:
            device: PyTorch device (for GPU sync)
            track_memory: Whether to track peak memory during step
            batch_size: Batch size for throughput calculation
        """
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.track_memory = track_memory
        self.batch_size = batch_size
        self.elapsed: float = 0.0
        self.peak_memory_gb: float = 0.0
        self._start_time: float = 0.0

    def __enter__(self) -> "StepTimer":
        if self.track_memory and self.device.type == "cuda":
            reset_peak_memory(self.device)
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        # Sync GPU before measuring time
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps" and torch.backends.mps.is_available():
            torch.mps.synchronize()

        self.elapsed = time.perf_counter() - self._start_time

        if self.track_memory:
            stats = get_memory_stats(self.device)
            self.peak_memory_gb = stats.peak_gb

    @property
    def metrics(self) -> StepMetrics:
        """Get metrics for this step."""
        throughput = self.batch_size / self.elapsed if self.elapsed > 0 else 0.0
        return StepMetrics(
            step_time=self.elapsed,
            peak_memory_gb=self.peak_memory_gb,
            batch_size=self.batch_size,
            throughput=throughput,
        )


@dataclass(frozen=True)
class Checkpoint:
    """A named memory/time checkpoint during training."""

    name: str
    timestamp: float
    memory: MemoryStats


@dataclass(frozen=True)
class TrainingProfiler:
    """Immutable profiler state for tracking memory and timing throughout training.

    Designed for explicit state threading in training loops:
    - mark() returns a new profiler state plus current memory stats
    - add_step() returns a new profiler state with an appended step record
    - sync(profiler) reduces only the unsynchronized suffix into global aggregates
    - current_metrics()/final_summary() read from explicit profiler state

    Example:
        >>> profiler = TrainingProfiler(torch.device("cuda"))
        >>> profiler, _ = profiler.mark("model_loaded")
        >>>
        >>> for step, batch in enumerate(dataloader):
        ...     timer = StepTimer("cuda", batch_size=len(batch))
        ...     with timer:
        ...         train_step(batch)
        ...     profiler = profiler.add_step(timer)
        ...
        ...     if step % 10 == 0:
        ...         # Log to console
        ...         print(profiler.step_summary())
        ...         # Log to WANDB
        ...         wandb.log(profiler.current_metrics())
        >>>
        >>> # Final summary
        >>> print(profiler.final_summary())
    """

    device: torch.device
    checkpoints: tuple[Checkpoint, ...] = field(default_factory=tuple)
    step_metrics: tuple[StepMetrics, ...] = field(default_factory=tuple)
    _synced_checkpoints: int = 0
    _synced_steps: int = 0
    _observed_peak_gb: float = 0.0
    _start_time: float = field(default_factory=time.perf_counter)

    @property
    def is_fully_synced(self) -> bool:
        """Whether there are no pending local records left to synchronize."""
        return self._synced_steps == len(
            self.step_metrics
        ) and self._synced_checkpoints == len(self.checkpoints)

    def mark(self, name: str) -> tuple["TrainingProfiler", MemoryStats]:
        """Record a named checkpoint and return updated profiler state.

        Args:
            name: Name for this checkpoint (e.g., "model_loaded", "after_warmup")

        Returns:
            Tuple of (updated profiler, current MemoryStats)

        Example:
            >>> profiler, _ = profiler.mark("model_loaded")
            >>> # ... training ...
            >>> profiler, _ = profiler.mark("training_complete")
        """
        stats = get_memory_stats(self.device)
        peak_gb = max(self._observed_peak_gb, stats.peak_gb)
        checkpoint = Checkpoint(
            name=name,
            timestamp=time.perf_counter() - self._start_time,
            memory=stats,
        )
        profiler = replace(
            self,
            checkpoints=self.checkpoints + (checkpoint,),
            _observed_peak_gb=peak_gb,
        )
        return profiler, stats

    def add_step(self, step: StepMetrics | StepTimer) -> "TrainingProfiler":
        """Return updated profiler state with a recorded step.

        Args:
            step: Completed step measurement or timer.

        Returns:
            Updated profiler state.

        Example:
            >>> timer = StepTimer("cuda", batch_size=32)
            >>> with timer:
            ...     train_step(batch)
            >>> profiler = profiler.add_step(timer)
        """
        metrics = step.metrics if isinstance(step, StepTimer) else step
        peak_gb = max(self._observed_peak_gb, metrics.peak_memory_gb)
        return replace(
            self,
            step_metrics=self.step_metrics + (metrics,),
            _observed_peak_gb=peak_gb,
        )

    @property
    def num_steps(self) -> int:
        """Number of training steps recorded."""
        return len(self.step_metrics)

    @property
    def step_times(self) -> tuple[float, ...]:
        """Step times in seconds."""
        return tuple(step.step_time for step in self.step_metrics)

    @property
    def step_peak_memories(self) -> tuple[float, ...]:
        """Peak memory for each recorded step."""
        return tuple(step.peak_memory_gb for step in self.step_metrics)

    @property
    def step_batch_sizes(self) -> tuple[int, ...]:
        """Batch size for each recorded step."""
        return tuple(step.batch_size for step in self.step_metrics)

    @property
    def total_time(self) -> float:
        """Total training time in seconds."""
        return sum(self.step_times)

    @property
    def avg_step_time(self) -> float:
        """Average step time in seconds."""
        if not self.step_metrics:
            return 0.0
        return self.total_time / self.num_steps

    @property
    def avg_step_time_stable(self) -> float:
        """Average step time excluding first step (warmup)."""
        if self.num_steps <= 1:
            return self.avg_step_time
        return sum(self.step_times[1:]) / (self.num_steps - 1)

    @property
    def peak_memory_gb(self) -> float:
        """Maximum peak memory across all steps."""
        current_peak = get_memory_stats(self.device).peak_gb
        step_peak = max(self.step_peak_memories) if self.step_metrics else 0.0
        return max(self._observed_peak_gb, step_peak, current_peak)

    @property
    def avg_throughput(self) -> float:
        """Average throughput in samples/second."""
        total_samples = sum(self.step_batch_sizes)
        if self.total_time > 0:
            return total_samples / self.total_time
        return 0.0

    def current_metrics(self, prefix: str = "") -> dict[str, float | bool]:
        """Get current metrics as dict for WANDB logging.

        Args:
            prefix: Optional prefix for all keys

        Returns:
            Dict with performance and memory metrics
        """
        if not self.step_metrics:
            return {}

        last_step = self.step_metrics[-1]

        mem = get_memory_stats(self.device)

        metrics = {
            "step_time_sec": last_step.step_time,
            "throughput_samples_sec": last_step.throughput,
            "avg_step_time_sec": self.avg_step_time_stable,
            "memory_allocated_gb": mem.allocated_gb,
            "memory_reserved_gb": mem.reserved_gb,
            "memory_peak_gb": self.peak_memory_gb,
            "memory_peak_exact": mem.exact_peak,
            "memory_reserved_exact": mem.exact_reserved,
            "memory_free_known": mem.known_free,
            "memory_total_known": mem.known_total,
        }

        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

        return metrics

    def step_summary(self) -> str:
        """One-line summary of recent step performance."""
        if not self.step_metrics:
            return "No steps recorded"

        last_step = self.step_metrics[-1]

        return (
            f"Step: {last_step.step_time:.2f}s | "
            f"Mem: {last_step.peak_memory_gb:.1f}GB | "
            f"Throughput: {last_step.throughput:.1f} samples/s"
        )

    def final_summary(self) -> str:
        """Comprehensive summary for end of training."""
        lines = [
            "=" * 60,
            "Training Performance Summary",
            "=" * 60,
        ]

        if self.num_steps > 0:
            lines.extend(
                [
                    f"Total steps:           {self.num_steps}",
                    f"Total time:            {self.total_time:.1f}s",
                    f"Avg step time:         {self.avg_step_time:.2f}s (with warmup)",
                    f"Avg step time:         {self.avg_step_time_stable:.2f}s (stable)",
                    f"Steps per minute:      {60.0 / self.avg_step_time_stable:.1f}",
                    f"Avg throughput:        {self.avg_throughput:.1f} samples/s",
                ]
            )

        mem = get_memory_stats(self.device)
        if mem.total_gb > 0:
            lines.extend(
                [
                    "",
                    "Memory:",
                    f"  Peak allocated:      {self.peak_memory_gb:.2f} GB",
                    f"  Current allocated:   {mem.allocated_gb:.2f} GB",
                    f"  Total GPU memory:    {mem.total_gb:.2f} GB",
                    f"  Utilization:         {self.peak_memory_gb / mem.total_gb:.1%}",
                ]
            )
        elif self.peak_memory_gb > 0:
            lines.extend(
                [
                    "",
                    "Memory:",
                    f"  Peak allocated:      {self.peak_memory_gb:.2f} GB",
                    "  Note: peak/free/total are approximate on this backend.",
                ]
            )

        if self.checkpoints:
            lines.extend(
                [
                    "",
                    "Checkpoints:",
                ]
            )
            for cp in self.checkpoints:
                lines.append(
                    f"  [{cp.timestamp:7.1f}s] {cp.name}: {cp.memory.peak_gb:.2f} GB peak"
                )

        lines.append("=" * 60)
        return "\n".join(lines)

    def checkpoint_summary(self) -> str:
        """Summary of memory at each checkpoint."""
        if not self.checkpoints:
            return "No checkpoints recorded"

        lines = ["Checkpoints:"]
        prev_mem = 0.0
        for cp in self.checkpoints:
            delta = cp.memory.peak_gb - prev_mem
            delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
            lines.append(
                f"  {cp.name:30s}: {cp.memory.peak_gb:6.2f} GB ({delta_str} GB)"
            )
            prev_mem = cp.memory.peak_gb
        return "\n".join(lines)


__all__ = [
    # New API
    "MemoryStats",
    "StepMetrics",
    "StepTimer",
    "TrainingProfiler",
    "Checkpoint",
    # Utility functions
    "get_memory_stats",
    "reset_peak_memory",
    "print_memory",
    "empty_cache",
]
