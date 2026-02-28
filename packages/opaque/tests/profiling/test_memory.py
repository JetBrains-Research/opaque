"""Tests for memory profiling tools.

Tests the profiling API (MemoryStats, StepTimer, TrainingProfiler)
across different devices (CUDA, MPS, CPU).
"""

import pytest
import torch

from opaque.profiling import (
    MemoryStats,
    StepMetrics,
    StepTimer,
    TrainingProfiler,
    get_memory_stats,
)


# Test on available devices
DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.insert(0, "cuda")
if torch.backends.mps.is_available():
    DEVICES.insert(0, "mps")


class TestMemoryStats:
    """Tests for MemoryStats dataclass."""

    def test_default_values(self):
        """Should have zero default values."""
        stats = MemoryStats()
        assert stats.allocated_gb == 0.0
        assert stats.reserved_gb == 0.0
        assert stats.peak_gb == 0.0

    def test_utilization_with_zero_total(self):
        """Should return 0 utilization when total is 0."""
        stats = MemoryStats(peak_gb=1.0, total_gb=0.0)
        assert stats.utilization == 0.0

    def test_utilization_calculation(self):
        """Should calculate utilization correctly."""
        stats = MemoryStats(peak_gb=40.0, total_gb=80.0)
        assert stats.utilization == pytest.approx(0.5)

    def test_to_dict(self):
        """Should convert to dict for WANDB."""
        stats = MemoryStats(
            allocated_gb=10.0,
            reserved_gb=15.0,
            peak_gb=12.0,
            free_gb=65.0,
            total_gb=80.0,
        )
        d = stats.to_dict()
        assert d["memory/allocated_gb"] == 10.0
        assert d["memory/peak_gb"] == 12.0


class TestGetMemoryStats:
    """Tests for get_memory_stats function."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_returns_memory_stats(self, device):
        """Should return MemoryStats object."""
        stats = get_memory_stats(device)
        assert isinstance(stats, MemoryStats)

    def test_cpu_returns_zeros(self):
        """CPU should return all zeros."""
        stats = get_memory_stats("cpu")
        assert stats.allocated_gb == 0.0
        assert stats.peak_gb == 0.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_returns_real_values(self):
        """CUDA should return real memory values."""
        x = torch.randn(1000, 1000, device="cuda")
        stats = get_memory_stats("cuda")
        assert stats.total_gb > 0
        assert stats.allocated_gb > 0
        del x
        torch.cuda.empty_cache()


class TestStepTimer:
    """Tests for StepTimer context manager."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_measures_time(self, device):
        """Should measure elapsed time."""
        timer = StepTimer(device, track_memory=False)
        with timer:
            x = torch.randn(100, 100)
            _ = x @ x.T
        assert timer.elapsed > 0

    def test_metrics_property(self):
        """Should expose metrics property."""
        timer = StepTimer("cpu", batch_size=32)
        with timer:
            pass
        metrics = timer.metrics
        assert isinstance(metrics, StepMetrics)
        assert metrics.batch_size == 32


class TestTrainingProfiler:
    """Tests for TrainingProfiler class."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_initialization(self, device):
        """Should initialize with device."""
        profiler = TrainingProfiler(device)
        assert profiler.device == torch.device(device)
        assert profiler.num_steps == 0

    @pytest.mark.parametrize("device", DEVICES)
    def test_mark_creates_checkpoint(self, device):
        """Should create checkpoint with mark()."""
        profiler = TrainingProfiler(device)
        stats = profiler.mark("test_point")
        assert len(profiler.checkpoints) == 1
        assert profiler.checkpoints[0].name == "test_point"
        assert isinstance(stats, MemoryStats)

    @pytest.mark.parametrize("device", DEVICES)
    def test_step_context_manager(self, device):
        """Should work as context manager."""
        profiler = TrainingProfiler(device)
        with profiler.step(batch_size=32):
            x = torch.randn(100, 100)
            _ = x @ x.T
        assert profiler.num_steps == 1
        assert profiler.step_times[0] > 0
        assert profiler.step_batch_sizes[0] == 32

    def test_avg_step_time_stable(self):
        """Should exclude first step for stable average."""
        profiler = TrainingProfiler("cpu")
        profiler.step_times = [10.0, 2.0, 2.0, 2.0]
        assert profiler.avg_step_time_stable == pytest.approx(2.0)

    @pytest.mark.parametrize("device", DEVICES)
    def test_current_metrics(self, device):
        """Should return current metrics dict."""
        profiler = TrainingProfiler(device)
        with profiler.step(batch_size=32):
            pass
        metrics = profiler.current_metrics()
        assert "step_time_sec" in metrics
        assert "memory_peak_gb" in metrics

    @pytest.mark.parametrize("device", DEVICES)
    def test_final_summary(self, device):
        """Should generate comprehensive summary."""
        profiler = TrainingProfiler(device)
        profiler.mark("start")
        for _ in range(3):
            with profiler.step(batch_size=16):
                pass
        profiler.mark("end")
        summary = profiler.final_summary()
        assert "Training Performance Summary" in summary
        assert "Total steps:" in summary
