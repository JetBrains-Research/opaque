"""Memory and timing profiling tools for DP training.

This module provides lightweight tools for tracking memory usage and timing
during differentially private training. All profiling state is modeled as
explicit immutable records so it can be threaded through training loops and
safely synchronized across distributed ranks.

Key Components:
    - MemoryStats: Dataclass for GPU memory statistics
    - StepPerf: Frozen record of a single step's performance
    - step_perf: Context manager that measures one training step
    - PerfState: Functional accumulator for DDP sync + stable averages
    - Utility functions: get_memory_stats, print_memory, reset_peak_memory

Example - Basic usage in training loop:
    >>> from opaque.profiling import step_perf, PerfState
    >>>
    >>> perf_state = PerfState(device=device)
    >>> for step, batch in enumerate(dataloader):
    ...     with step_perf(device, batch_size=len(batch)) as perf:
    ...         grads = compute_gradients(batch)
    ...         perf.mark("clip")
    ...         update_params(grads)
    ...         perf.mark("update")
    ...     perf_state = perf_state.add(perf.result)
    ...     wandb.log({
    ...         "train/loss": loss,
    ...         **perf.result.to_dict(prefix="train/"),
    ...     })

Example - Simple memory tracking:
    >>> from opaque.profiling import get_memory_stats, print_memory
    >>>
    >>> print_memory(device, "After model load")
    >>> stats = get_memory_stats(device)
    >>> print(f"Peak: {stats.peak_gb:.2f} GB")
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import torch


@dataclass(frozen=True)
class MemoryStats:
    """GPU memory statistics at a point in time.

    All values are in GB for easy reading. Use to_dict() for logging.

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
        """Convert to dict for logging.

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


def _sync_device(device: torch.device) -> None:
    """Synchronize device for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


@dataclass(frozen=True)
class StepPerf:
    """Immutable record of a single step's performance.

    Produced by :func:`step_perf`. Contains wall-clock timing, throughput,
    memory statistics, and optional sub-step marks. All fields are frozen
    at the end of the ``step_perf`` context manager — no live hardware
    queries on read.

    Attributes:
        step_time_sec: Total wall-clock time for the step.
        samples_per_second: Throughput (batch_size / step_time_sec).
        steps_per_second: Inverse of step_time_sec.
        memory_peak_gb: Peak GPU memory during the step.
        memory_allocated_gb: Allocated memory at end of step.
        memory_reserved_gb: Reserved memory at end of step.
        batch_size: Number of samples processed.
        marks: Sub-step timing marks as ``{name: elapsed_sec}``.
    """

    step_time_sec: float = 0.0
    samples_per_second: float = 0.0
    steps_per_second: float = 0.0
    memory_peak_gb: float = 0.0
    memory_allocated_gb: float = 0.0
    memory_reserved_gb: float = 0.0
    batch_size: int = 0
    marks: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        """Convert to flat dict for logging.

        Returns bare keys by default — caller adds the prefix to control
        namespace (``train/``, ``eval/``, ``perf/``, etc.).

        Args:
            prefix: Optional prefix for all keys (e.g. ``"train/"``).

        Returns:
            Dict with performance metrics.
        """
        d: dict[str, float] = {
            f"{prefix}step_time_sec": self.step_time_sec,
            f"{prefix}samples_per_second": self.samples_per_second,
            f"{prefix}steps_per_second": self.steps_per_second,
            f"{prefix}memory_peak_gb": self.memory_peak_gb,
            f"{prefix}memory_allocated_gb": self.memory_allocated_gb,
            f"{prefix}memory_reserved_gb": self.memory_reserved_gb,
        }
        for name, elapsed in self.marks.items():
            d[f"{prefix}{name}_sec"] = elapsed
        return d


class _StepPerfBuilder:
    """Mutable builder used inside the ``step_perf`` context manager.

    Collects sub-step ``.mark()`` calls and is finalized into an
    immutable :class:`StepPerf` when the context exits.
    """

    def __init__(self, device: torch.device, batch_size: int) -> None:
        self._device = device
        self._batch_size = batch_size
        self._marks: dict[str, float] = {}
        self._last_mark_time: float = 0.0
        self._result: StepPerf | None = None

    def mark(self, name: str) -> None:
        """Record a sub-step timing mark.

        Each mark records the elapsed time since the previous mark (or
        since the step started if this is the first mark). The caller
        controls whether to insert a GPU sync before the mark for
        accurate sub-step timing on accelerators.

        Args:
            name: Label for this sub-step (e.g. ``"clip"``, ``"noise"``,
                ``"optimizer"``).
        """
        now = time.perf_counter()
        self._marks[name] = now - self._last_mark_time
        self._last_mark_time = now

    @property
    def result(self) -> StepPerf:
        """The finalized :class:`StepPerf` record.

        Only available after the ``step_perf`` context manager exits.

        Raises:
            RuntimeError: If accessed before the context manager exits.
        """
        if self._result is None:
            raise RuntimeError(
                "StepPerf result is not available until the step_perf "
                "context manager exits."
            )
        return self._result


@contextmanager
def step_perf(
    device: torch.device | str,
    *,
    batch_size: int = 0,
    track_memory: bool = True,
) -> Iterator[_StepPerfBuilder]:
    """Time a training step and produce an immutable :class:`StepPerf` record.

    Handles GPU synchronization for accurate timing and optional peak
    memory tracking. Supports sub-step ``.mark()`` calls for
    fine-grained breakdowns.

    Args:
        device: PyTorch device (for GPU sync and memory reads).
        batch_size: Number of samples in this step (for throughput).
        track_memory: Whether to record peak memory (default True).

    Yields:
        A :class:`_StepPerfBuilder` that supports ``.mark(name)``
        during the step body. After the context exits, access the
        frozen record via ``.result``.

    Example:
        >>> with step_perf("cuda", batch_size=32) as perf:
        ...     grads = compute_gradients(batch)
        ...     perf.mark("clip")
        ...     update_params(grads)
        ...     perf.mark("update")
        >>> wandb.log(perf.result.to_dict(prefix="train/"))
    """
    if isinstance(device, str):
        device = torch.device(device)

    builder = _StepPerfBuilder(device, batch_size)

    if track_memory and device.type == "cuda":
        reset_peak_memory(device)

    _sync_device(device)
    t0 = time.perf_counter()
    builder._last_mark_time = t0

    yield builder

    _sync_device(device)
    elapsed = time.perf_counter() - t0

    mem = get_memory_stats(device) if track_memory else MemoryStats()

    builder._result = StepPerf(
        step_time_sec=elapsed,
        samples_per_second=batch_size / elapsed if elapsed > 0 else 0.0,
        steps_per_second=1.0 / elapsed if elapsed > 0 else 0.0,
        memory_peak_gb=mem.peak_gb,
        memory_allocated_gb=mem.allocated_gb,
        memory_reserved_gb=mem.reserved_gb,
        batch_size=batch_size,
        marks=MappingProxyType(dict(builder._marks)),
    )


@dataclass(frozen=True)
class PerfState:
    """Functional accumulator for step performance across a training run.

    Tracks cumulative timing and throughput statistics with
    warmup-excluded stable averages. Designed for DDP
    :func:`~opaque.distributed.sync` reduction.

    All fields are frozen — call :meth:`add` to produce a new state
    with updated counters.

    Attributes:
        device: PyTorch device (carried for DDP sync).
        num_steps: Total number of recorded steps.
        total_time: Cumulative wall-clock time across all steps.
        total_samples: Cumulative sample count.
        num_steps_stable: Steps after warmup (excludes first step).
        total_time_stable: Cumulative time for stable steps.
        total_samples_stable: Cumulative samples for stable steps.
        max_peak_memory_gb: High-water mark of peak memory.
        last_step: Most recent :class:`StepPerf` record.
    """

    device: torch.device
    num_steps: int = 0
    total_time: float = 0.0
    total_samples: int = 0
    num_steps_stable: int = 0
    total_time_stable: float = 0.0
    total_samples_stable: int = 0
    max_peak_memory_gb: float = 0.0
    last_step: StepPerf | None = None

    def add(self, perf: StepPerf) -> "PerfState":
        """Return a new state incorporating a completed step.

        The first step is treated as warmup and excluded from the
        ``*_stable`` counters.

        Args:
            perf: Completed step record.

        Returns:
            Updated PerfState.
        """
        num_steps = self.num_steps + 1
        total_time = self.total_time + perf.step_time_sec
        total_samples = self.total_samples + perf.batch_size
        max_peak = max(self.max_peak_memory_gb, perf.memory_peak_gb)

        num_steps_stable = self.num_steps_stable
        total_time_stable = self.total_time_stable
        total_samples_stable = self.total_samples_stable
        if self.num_steps >= 1:
            num_steps_stable += 1
            total_time_stable += perf.step_time_sec
            total_samples_stable += perf.batch_size

        return replace(
            self,
            num_steps=num_steps,
            total_time=total_time,
            total_samples=total_samples,
            max_peak_memory_gb=max_peak,
            num_steps_stable=num_steps_stable,
            total_time_stable=total_time_stable,
            total_samples_stable=total_samples_stable,
            last_step=perf,
        )

    @property
    def avg_step_time(self) -> float:
        """Average step time in seconds (all steps)."""
        if self.num_steps == 0:
            return 0.0
        return self.total_time / self.num_steps

    @property
    def avg_step_time_stable(self) -> float:
        """Average step time excluding first step (warmup)."""
        if self.num_steps_stable == 0:
            return self.avg_step_time
        return self.total_time_stable / self.num_steps_stable

    @property
    def avg_samples_per_second(self) -> float:
        """Average throughput in samples/second (all steps)."""
        if self.total_time > 0:
            return self.total_samples / self.total_time
        return 0.0

    @property
    def avg_samples_per_second_stable(self) -> float:
        """Average throughput in samples/second (stable, excluding warmup)."""
        if self.total_time_stable > 0:
            return self.total_samples_stable / self.total_time_stable
        return self.avg_samples_per_second

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        """Accumulated metrics as flat dict for logging.

        Args:
            prefix: Optional prefix for all keys.

        Returns:
            Dict with running averages and peak memory.
        """
        return {
            f"{prefix}avg_step_time_sec": self.avg_step_time_stable,
            f"{prefix}avg_samples_per_second": self.avg_samples_per_second_stable,
            f"{prefix}max_peak_memory_gb": self.max_peak_memory_gb,
        }


__all__ = [
    "StepPerf",
    "step_perf",
    "PerfState",
    "MemoryStats",
    "get_memory_stats",
    "reset_peak_memory",
    "print_memory",
    "empty_cache",
]
