"""Profiling façade — re-exports from ``opaque.api.engine.profiling``."""

from opaque.api.engine.profiling import (
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
