"""TrainingProfiler sync under NCCL."""

from __future__ import annotations

import pytest
import torch

from opaque.api.engine.distributed._state import reduce_scalar
from opaque.distributed import sync
from opaque.profiling import StepTimer, TrainingProfiler

from ._ddp_helpers import _cleanup_ddp, _setup_ddp, _spawn


pytestmark = pytest.mark.cuda


def _worker_sync_profiler(rank: int, world_size: int, port: int) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        profiler = TrainingProfiler(device)

        timer = StepTimer(device, batch_size=4)
        with timer:
            x = torch.randn(1024 + 512 * rank, device=device)
            _ = (x * x).sum()
        profiler = profiler.add_step(timer)

        synced_profiler = sync(profiler)
        assert synced_profiler is not profiler
        assert synced_profiler.num_steps == 1
        assert synced_profiler._step_batch_sizes[-1] == world_size * 4
        assert profiler._step_batch_sizes[-1] == 4
        assert synced_profiler.is_fully_synced
        assert synced_profiler._step_metrics[-1]._step_time >= 0.0

        local_peak = float(synced_profiler._observed_peak_gb)
        peak_min = reduce_scalar(local_peak, op="min", device=device)
        peak_max = reduce_scalar(local_peak, op="max", device=device)
        assert abs(peak_max - peak_min) < 1e-6

        twice_synced = sync(synced_profiler)
        assert twice_synced is synced_profiler

        profiler_with_mark, _ = synced_profiler.mark("after_sync")
        assert not profiler_with_mark.is_fully_synced
        synced_with_mark = sync(profiler_with_mark)
        assert synced_with_mark.is_fully_synced
        assert len(synced_with_mark.checkpoints) == 1
        assert synced_with_mark.checkpoints[0].name == "after_sync"
        assert synced_with_mark._synced_checkpoints == len(synced_with_mark.checkpoints)
    finally:
        _cleanup_ddp()


class TestProfilerSyncDistributed:
    def test_sync_profiler(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_sync_profiler)
