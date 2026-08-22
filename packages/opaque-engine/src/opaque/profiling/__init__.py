"""Performance profiling for DP training.

Factory functions live here; the records and trackers they return —
``MemoryStats``, ``PerfStage``, ``PerfState``, ``PerfTracker``,
``StepPerf`` — live in :mod:`opaque.profiling.types` for ``isinstance``
checks and type annotations, matching :mod:`opaque.scheduling` and
:mod:`opaque.optimizers`.
"""

from opaque.api.engine.profiling import (
    empty_cache,
    get_memory_stats,
    perf_state,
    perf_tracker,
    print_memory,
    reset_peak_memory,
    step_perf,
)
from opaque.profiling import types

__all__ = [
    "empty_cache",
    "get_memory_stats",
    "perf_state",
    "perf_tracker",
    "print_memory",
    "reset_peak_memory",
    "step_perf",
    "types",
]
