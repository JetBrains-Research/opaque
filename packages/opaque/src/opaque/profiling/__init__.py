"""Memory profiling and performance diagnostics for DP training.

This module provides tools for tracking memory usage and timing during
differentially private training.

Main Components:
    - TrainingProfiler: Full-featured profiler for training loops
    - StepTimer: Context manager for timing individual steps
    - MemoryStats: Dataclass for memory statistics
    - Utility functions: get_memory_stats, print_memory, reset_peak_memory

Example:
    >>> from opaque.profiling import TrainingProfiler, print_memory
    >>>
    >>> # Simple memory check
    >>> print_memory(device, "After model load")
    >>>
    >>> # Full profiling
    >>> profiler = TrainingProfiler(device)
    >>> profiler.mark("model_loaded")
    >>> for batch in dataloader:
    ...     with profiler.step(batch_size=len(batch)):
    ...         train_step(batch)
    >>> print(profiler.final_summary())
"""

from opaque.profiling.memory import (
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
