"""Distributed synchronization helpers for profiling components."""

from __future__ import annotations

from dataclasses import replace

import torch

from opaque.distributed import (
    assert_scalar_equal,
    is_distributed,
    reduce_scalar,
    register_sync_type,
)

from .memory import Checkpoint, StepMetrics, TrainingProfiler

__all__ = ["sync_training_profiler"]


def sync_training_profiler(profiler: TrainingProfiler) -> TrainingProfiler:
    """Synchronize pending profiler records across distributed ranks.

    This helper aggregates only the pending local suffix of the profiler:
    - step time uses max across ranks (critical path)
    - batch size uses sum across ranks (global samples)
    - peak memory uses max across ranks

    Calling ``sync(profiler)`` multiple times is safe because synchronized
    records are moved into the synced prefix of the immutable profiler state.
    """
    if not is_distributed():
        return profiler

    if profiler.is_fully_synced:
        return profiler

    device = profiler.device if isinstance(profiler.device, torch.device) else torch.device(profiler.device)

    assert_scalar_equal(
        float(len(profiler.pending_steps)),
        name="TrainingProfiler.pending_steps",
        device=device,
    )
    assert_scalar_equal(
        float(len(profiler.pending_checkpoints)),
        name="TrainingProfiler.pending_checkpoints",
        device=device,
    )

    synced_steps = []
    for step in profiler.pending_steps:
        step_time = reduce_scalar(float(step.step_time), op="max", device=device)
        batch_size = int(
            reduce_scalar(float(step.batch_size), op="sum", device=device)
        )
        peak_memory_gb = reduce_scalar(
            float(step.peak_memory_gb), op="max", device=device
        )
        throughput = batch_size / step_time if step_time > 0 else 0.0
        synced_steps.append(
            StepMetrics(
                step_time=step_time,
                peak_memory_gb=peak_memory_gb,
                batch_size=batch_size,
                throughput=throughput,
            )
        )

    synced_checkpoints = []
    for checkpoint in profiler.pending_checkpoints:
        timestamp = reduce_scalar(float(checkpoint.timestamp), op="max", device=device)
        peak_gb = reduce_scalar(
            float(checkpoint.memory.peak_gb), op="max", device=device
        )
        synced_checkpoints.append(
            Checkpoint(
                name=checkpoint.name,
                timestamp=timestamp,
                memory=replace(checkpoint.memory, peak_gb=peak_gb),
            )
        )

    observed_peak_gb = reduce_scalar(
        float(profiler._observed_peak_gb), op="max", device=device
    )

    return replace(
        profiler,
        synced_steps=profiler.synced_steps + tuple(synced_steps),
        pending_steps=(),
        synced_checkpoints=profiler.synced_checkpoints + tuple(synced_checkpoints),
        pending_checkpoints=(),
        _observed_peak_gb=observed_peak_gb,
    )


register_sync_type(TrainingProfiler, sync_training_profiler)
