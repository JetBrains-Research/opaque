"""Distributed synchronization helpers for profiling components."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import torch

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

    device = (
        state.device
        if isinstance(state.device, torch.device)
        else torch.device(state.device)
    )

    assert_scalar_equal(
        state.num_steps,
        name="PerfState.num_steps",
        device=device,
    )

    total_time = reduce_scalar(float(state.total_time), op="max", device=device)
    total_samples = int(reduce_scalar(state.total_samples, op="sum", device=device))
    max_peak = reduce_scalar(float(state.max_peak_memory_gb), op="max", device=device)

    total_time_stable = reduce_scalar(
        float(state.total_time_stable), op="max", device=device
    )
    total_samples_stable = int(
        reduce_scalar(state.total_samples_stable, op="sum", device=device)
    )

    last_step = state.last_step
    if last_step is not None:
        last_time = reduce_scalar(
            float(last_step.step_time_sec), op="max", device=device
        )
        last_samples = int(reduce_scalar(last_step.batch_size, op="sum", device=device))
        last_peak = reduce_scalar(
            float(last_step.memory_peak_gb), op="max", device=device
        )
        last_step = StepPerf(
            step_time_sec=last_time,
            samples_per_second=last_samples / last_time if last_time > 0 else 0.0,
            memory_peak_gb=last_peak,
            memory_allocated_gb=0.0,
            memory_reserved_gb=0.0,
            batch_size=last_samples,
            marks=MappingProxyType({}),
        )

    return replace(
        state,
        total_time=total_time,
        total_samples=total_samples,
        max_peak_memory_gb=max_peak,
        total_time_stable=total_time_stable,
        total_samples_stable=total_samples_stable,
        last_step=last_step,
    )


def _sync_last(last: StepPerf | None, device: torch.device) -> StepPerf | None:
    if last is None:
        return None
    last_time = reduce_scalar(float(last.step_time_sec), op="max", device=device)
    last_samples = int(reduce_scalar(last.batch_size, op="sum", device=device))
    last_peak = reduce_scalar(float(last.memory_peak_gb), op="max", device=device)
    return StepPerf(
        step_time_sec=last_time,
        samples_per_second=last_samples / last_time if last_time > 0 else 0.0,
        memory_peak_gb=last_peak,
        memory_allocated_gb=0.0,
        memory_reserved_gb=0.0,
        batch_size=last_samples,
        marks=MappingProxyType({}),
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

    for name, stage in tracker._stages.items():
        assert_scalar_equal(
            stage.num_steps,
            name=f"PerfStage({name!r}).num_steps",
            device=device,
        )

        s = synced._get_stage(name)
        s.num_steps = stage.num_steps
        s._warmup_steps = stage._warmup_steps
        s.total_time = reduce_scalar(float(stage.total_time), op="max", device=device)
        s.total_samples = int(
            reduce_scalar(stage.total_samples, op="sum", device=device)
        )
        s.max_peak_memory_gb = reduce_scalar(
            float(stage.max_peak_memory_gb), op="max", device=device
        )
        s.last = _sync_last(stage.last, device)

    return synced


register_sync_type(PerfState, sync_perf_state)
register_sync_type(PerfTracker, sync_perf_tracker)
