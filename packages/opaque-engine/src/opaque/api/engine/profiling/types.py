"""Public type definitions for :mod:`opaque.profiling`.

Re-exports the pure data records (``MemoryStats``, ``StepMetrics``,
``Checkpoint``) for type annotations. The interactive profiler classes
(``TrainingProfiler``, ``StepTimer``) live in the package init.
"""

from __future__ import annotations

from opaque.api.engine.profiling._memory import Checkpoint, MemoryStats, StepMetrics

__all__ = ["MemoryStats", "StepMetrics", "Checkpoint"]
