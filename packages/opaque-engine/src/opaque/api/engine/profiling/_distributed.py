"""Distributed synchronization helpers for profiling components."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Any

from opaque.api.engine import runtime
from opaque.api.engine.distributed import is_distributed
from opaque.api.engine.distributed._state import (
    assert_scalar_equal,
    reduce_scalar,
    register_sync_type,
)

from ._memory import (
    PerfState,
    PerfTracker,
    StepPerf,
)

__all__ = ["sync_perf_state", "sync_perf_tracker"]


def _sync_step_perf(last: StepPerf | None, device: Any) -> StepPerf | None:
    """Synchronize an optional step record with a fixed presence schedule."""
    local_presence = int(last is not None)
    min_presence = reduce_scalar(local_presence, op="min")
    max_presence = reduce_scalar(local_presence, op="max")
    if min_presence != max_presence:
        raise RuntimeError("StepPerf presence mismatch across ranks.")
    if not local_presence:
        return None

    assert last is not None
    last_time = reduce_scalar(float(last.step_time_sec), op="max")
    last_samples = int(reduce_scalar(float(last.batch_size), op="sum"))
    last_peak = _reduce_optional(last.memory_peak_gb, op="max")
    return StepPerf(
        step_time_sec=last_time,
        samples_per_second=last_samples / last_time if last_time > 0 else 0.0,
        memory_peak_gb=last_peak,
        memory_allocated_gb=None,
        memory_reserved_gb=None,
        batch_size=last_samples,
        marks=MappingProxyType({}),
    )


def _reduce_optional(value: float | None, *, op: str) -> float | None:
    """Reduce a metric only when every rank reports it."""
    present = int(value is not None)
    all_present = reduce_scalar(present, op="min")
    if not all_present:
        return None
    assert value is not None
    return float(reduce_scalar(value, op=op))


def _synchronized_stage_names(tracker: PerfTracker) -> tuple[str, ...]:
    """Return the shared stage schema or raise before stage reductions begin."""
    local_names = tuple(sorted(tracker._stages))
    gathered = runtime.distributed_all_gather_object(local_names)
    mismatched = [rank for rank, names in enumerate(gathered) if names != local_names]
    if mismatched:
        raise RuntimeError(
            "PerfTracker stage schema mismatch across ranks: "
            f"mismatched ranks={mismatched}."
        )
    return local_names


def sync_perf_state(state: PerfState) -> PerfState:
    """Synchronize PerfState across distributed ranks.

    - step time uses max across ranks (critical path)
    - batch size / samples uses sum across ranks (global samples)
    - peak memory uses max across ranks

    Safe to call multiple times or on non-distributed runs (returns
    the input unchanged).
    """
    if not is_distributed():
        return state

    device = state.device

    assert_scalar_equal(
        state.num_steps,
        name="PerfState.num_steps",
    )

    total_time = reduce_scalar(float(state.total_time), op="max")
    total_samples = int(reduce_scalar(state.total_samples, op="sum"))
    max_peak = _reduce_optional(state.max_peak_memory_gb, op="max")

    total_time_stable = reduce_scalar(float(state.total_time_stable), op="max")
    total_samples_stable = int(reduce_scalar(state.total_samples_stable, op="sum"))

    last_step = _sync_step_perf(state.last_step, device)

    return replace(
        state,
        total_time=total_time,
        total_samples=total_samples,
        max_peak_memory_gb=max_peak,
        total_time_stable=total_time_stable,
        total_samples_stable=total_samples_stable,
        last_step=last_step,
    )


def sync_perf_tracker(tracker: PerfTracker) -> PerfTracker:
    """Synchronize PerfTracker across distributed ranks.

    Returns a new :class:`PerfTracker` with reduced per-stage values.
    The original tracker is not modified.
    """
    if not is_distributed():
        return tracker

    device = tracker.device
    synced = PerfTracker(device, tracker._warmup_steps)

    for name in _synchronized_stage_names(tracker):
        stage = tracker._stages[name]
        assert_scalar_equal(
            stage.num_steps,
            name=f"PerfStage({name!r}).num_steps",
        )

        s = synced._get_stage(name)
        s.num_steps = stage.num_steps
        s._warmup_steps = stage._warmup_steps
        s.total_time = reduce_scalar(float(stage.total_time), op="max")
        s.total_samples = int(reduce_scalar(stage.total_samples, op="sum"))
        s.max_peak_memory_gb = _reduce_optional(stage.max_peak_memory_gb, op="max")
        s.last = _sync_step_perf(stage.last, device)

    return synced


register_sync_type(PerfState, sync_perf_state)
register_sync_type(PerfTracker, sync_perf_tracker)
