"""Memory profiling and performance diagnostics for DP training.

Provides stateful profiler classes plus convenience helpers for tracking
memory usage and timing during differentially private training.

- :class:`TrainingProfiler` — immutable profiler state for training loops
- :class:`StepTimer` — context manager for timing an individual step
- :func:`get_memory_stats`, :func:`print_memory`,
  :func:`reset_peak_memory`, :func:`empty_cache` — point-in-time helpers

Pure data records (``MemoryStats``, ``StepMetrics``, ``Checkpoint``) live
in :mod:`opaque.profiling.types`.

Example:
    >>> from opaque.profiling import StepTimer, TrainingProfiler, print_memory
    >>>
    >>> print_memory(device, "After model load")
    >>> profiler = TrainingProfiler(device)
    >>> profiler, _ = profiler.mark("model_loaded")
    >>> for batch in dataloader:
    ...     timer = StepTimer(device, batch_size=len(batch))
    ...     with timer:
    ...         train_step(batch)
    ...     profiler = profiler.add_step(timer)
    >>> print(profiler.final_summary())
"""

from opaque.profiling._memory import (
    StepTimer,
    TrainingProfiler,
    empty_cache,
    get_memory_stats,
    print_memory,
    reset_peak_memory,
)

__all__ = [
    "TrainingProfiler",
    "StepTimer",
    "get_memory_stats",
    "print_memory",
    "reset_peak_memory",
    "empty_cache",
]
