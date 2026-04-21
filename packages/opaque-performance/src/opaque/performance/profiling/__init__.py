"""Memory profiling and performance diagnostics for DP training.

This module provides tools for tracking memory usage and timing during
differentially private training with explicit profiler state.

Main Components:
    - TrainingProfiler: Immutable profiler state for training loops
    - StepTimer: Context manager for timing an individual step
    - MemoryStats: Dataclass for memory statistics
    - Utility functions: get_memory_stats, print_memory, reset_peak_memory

Example:
    >>> from opaque.performance.profiling import StepTimer, TrainingProfiler, print_memory
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

from opaque.performance.profiling.memory import (
    Checkpoint,
    MemoryStats,
    StepMetrics,
    StepTimer,
    TrainingProfiler,
    empty_cache,
    get_memory_stats,
    print_memory,
    reset_peak_memory,
)

__all__ = [
    # Main classes
    "TrainingProfiler",
    "StepTimer",
    "MemoryStats",
    "StepMetrics",
    "Checkpoint",
    # Utility functions
    "get_memory_stats",
    "print_memory",
    "reset_peak_memory",
    "empty_cache",
]
