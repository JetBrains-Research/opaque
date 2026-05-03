"""Tests for memory profiling tools.

Tests the profiling API (MemoryStats, StepTimer, TrainingProfiler)
across different devices (CUDA, MPS, CPU).
"""

import pytest
import torch

from opaque.core.profiling import (
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
        assert stats.exact_peak is False
        assert stats.known_total is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_returns_real_values(self):
        """CUDA should return real memory values."""
        x = torch.randn(1000, 1000, device="cuda")
        stats = get_memory_stats("cuda")
        assert stats.total_gb > 0
        assert stats.allocated_gb > 0
        assert stats.exact_peak is True
        assert stats.exact_reserved is True
        assert stats.known_total is True
        assert stats.known_free is True
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

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    def test_mps_synchronizes_before_timing(self, monkeypatch):
        """MPS timer should synchronize before measuring elapsed time."""
        calls = {"count": 0}

        def _sync() -> None:
            calls["count"] += 1

        monkeypatch.setattr(torch.mps, "synchronize", _sync)

        timer = StepTimer("mps", track_memory=False)
        with timer:
            x = torch.randn(16, 16, device="mps")
            _ = x @ x.T

        assert calls["count"] == 1


class TestTrainingProfiler:
    """Tests for TrainingProfiler class."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_initialization(self, device):
        """Should initialize with device."""
        profiler = TrainingProfiler(torch.device(device))
        assert profiler.device == torch.device(device)
        assert profiler.num_steps == 0

    @pytest.mark.parametrize("device", DEVICES)
    def test_mark_creates_checkpoint(self, device):
        """Should create checkpoint with mark()."""
        profiler = TrainingProfiler(torch.device(device))
        profiler, stats = profiler.mark("test_point")
        assert len(profiler.checkpoints) == 1
        assert profiler.checkpoints[0].name == "test_point"
        assert isinstance(stats, MemoryStats)

    @pytest.mark.parametrize("device", DEVICES)
    def test_add_step(self, device):
        """Should record completed step metrics."""
        profiler = TrainingProfiler(torch.device(device))
        timer = StepTimer(device, batch_size=32)
        with timer:
            x = torch.randn(100, 100)
            _ = x @ x.T
        profiler = profiler.add_step(timer)
        assert profiler.num_steps == 1
        assert profiler.step_times[0] > 0
        assert profiler.step_batch_sizes[0] == 32

    def test_avg_step_time_stable(self):
        """Should exclude first step for stable average."""
        profiler = TrainingProfiler(
            torch.device("cpu"),
            step_metrics=(
                StepMetrics(step_time=10.0, batch_size=1, throughput=0.1),
                StepMetrics(step_time=2.0, batch_size=1, throughput=0.5),
                StepMetrics(step_time=2.0, batch_size=1, throughput=0.5),
                StepMetrics(step_time=2.0, batch_size=1, throughput=0.5),
            ),
        )
        assert profiler.avg_step_time_stable == pytest.approx(2.0)

    @pytest.mark.parametrize("device", DEVICES)
    def test_current_metrics(self, device):
        """Should return current metrics dict."""
        profiler = TrainingProfiler(torch.device(device))
        timer = StepTimer(device, batch_size=32)
        with timer:
            pass
        profiler = profiler.add_step(timer)
        metrics = profiler.current_metrics()
        assert "step_time_sec" in metrics
        assert "memory_peak_gb" in metrics
        assert "memory_peak_exact" in metrics
        assert "memory_reserved_exact" in metrics
        assert "memory_total_known" in metrics

    def test_software_peak_tracking_from_checkpoints(self, monkeypatch):
        """Profiler should keep high-water peak from checkpoints."""
        profiler = TrainingProfiler(torch.device("cpu"))

        peaks = iter([0.10, 0.35, 0.20])

        def fake_stats(_device):
            peak = next(peaks)
            return MemoryStats(peak_gb=peak)

        monkeypatch.setattr(
            "opaque.core.profiling.memory.get_memory_stats", fake_stats
        )

        profiler, _ = profiler.mark("a")
        profiler, _ = profiler.mark("b")
        assert profiler.peak_memory_gb == pytest.approx(0.35)

    @pytest.mark.parametrize("device", DEVICES)
    def test_final_summary(self, device):
        """Should generate comprehensive summary."""
        profiler = TrainingProfiler(torch.device(device))
        profiler, _ = profiler.mark("start")
        for _ in range(3):
            timer = StepTimer(device, batch_size=16)
            with timer:
                pass
            profiler = profiler.add_step(timer)
        profiler, _ = profiler.mark("end")
        summary = profiler.final_summary()
        assert "Training Performance Summary" in summary
        assert "Total steps:" in summary
