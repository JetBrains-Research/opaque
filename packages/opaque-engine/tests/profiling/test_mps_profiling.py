"""MPS-specific profiling behavior (validated empirically on Apple Silicon).

These tests lock in the Apple-Silicon profiling guarantees that Track-B kernel
benchmarking depends on:

- memory stats report a real total / reserved budget (not zeros);
- ``peak_gb`` is the driver's reserved high-water, so it *captures transients*
  freed before the read — at *run* scope: ``step_perf`` never resets it on
  MPS, so ``StepPerf.peak_is_per_step`` is False there and the figure must
  be attributed to the run, not to the step;
- ``reset_peak_memory`` actually lowers that high-water on MPS;
- ``step_perf`` does NOT pay an ``empty_cache`` per step (it would tank a
  training loop), so MPS peak accumulates as the run high-water instead;
- ``.mark()`` is device-synchronized, so sub-step timings reflect real GPU
  execution rather than async kernel-launch time.

All are ``@pytest.mark.mps`` and auto-skip off Apple Silicon.
"""

import pytest
import torch

from opaque.profiling import (
    get_memory_stats,
    reset_peak_memory,
    step_perf,
)

_GiB_FLOATS = 256 * 1024 * 1024  # 256M float32 = 1 GiB


@pytest.mark.mps
class TestMpsMemoryStats:
    def test_total_and_reserved_are_real(self):
        stats = get_memory_stats("mps")
        # recommended_max_memory() → a real multi-GB budget, not the old 0.0.
        assert stats.total_gb > 1.0
        assert stats.known_total is True
        assert stats.reserved_gb >= stats.allocated_gb
        assert stats.exact_reserved is True
        # The driver reserved high-water is a precise measurement (it just
        # differs from CUDA's peak in quantity, not precision) → exact_peak.
        assert stats.exact_peak is True
        assert stats.peak_gb == stats.reserved_gb

    def test_allocated_reflects_a_known_tensor(self):
        before = get_memory_stats("mps").allocated_gb
        t = torch.empty(_GiB_FLOATS, dtype=torch.float32, device="mps")  # 1 GiB
        torch.mps.synchronize()
        after = get_memory_stats("mps").allocated_gb
        assert after - before == pytest.approx(1.0, abs=0.05)
        del t


@pytest.mark.mps
class TestMpsPeakTracking:
    def test_peak_captures_a_freed_transient(self):
        """The regression guard: peak must reflect a transient freed mid-step.

        Current allocation would report ~0 after the free; the driver
        high-water keeps the 1 GiB.  The number answers "how much did the
        run need so far", not "what this step alone needed" — see
        ``peak_is_per_step`` below.
        """
        reset_peak_memory("mps")  # clean high-water baseline
        with step_perf("mps", batch_size=1) as sp:
            t = torch.empty(_GiB_FLOATS, dtype=torch.float32, device="mps")  # 1 GiB
            torch.mps.synchronize()
            del t  # freed before the step ends
            torch.mps.synchronize()
        perf = sp.perf
        assert perf.memory_peak_gb >= 0.9  # captured the ~1 GiB transient
        # ... and is well above the (near-zero) end-of-step allocation.
        assert perf.memory_peak_gb > perf.memory_allocated_gb + 0.5

    def test_peak_is_not_per_step_on_mps(self):
        """peak_gb is the run-wide reserved high-water, not a step figure."""
        with step_perf("mps", batch_size=1) as sp:
            torch.randn(64, 64, device="mps")
        perf = sp.perf
        assert perf.peak_is_per_step is False
        assert perf.to_dict(prefix="train/")["train/peak_is_per_step"] is False
        # On MPS the reported peak equals the end-of-step reserved high-water.
        assert perf.memory_peak_gb == perf.memory_reserved_gb

    def test_reset_peak_memory_lowers_high_water(self):
        t = torch.empty(_GiB_FLOATS, dtype=torch.float32, device="mps")
        del t  # grows then frees → driver retains the high-water
        torch.mps.synchronize()
        high = get_memory_stats("mps").peak_gb
        reset_peak_memory("mps")  # empty_cache → re-baseline
        low = get_memory_stats("mps").peak_gb
        assert low < high

    def test_step_perf_does_not_empty_cache_per_step(self, monkeypatch):
        """empty_cache every step would cripple a training loop — guard it."""
        calls: list[int] = []
        monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append(1))
        with step_perf("mps", batch_size=1, track_memory=True):
            a = torch.randn(64, 64, device="mps")
            _ = a @ a
        assert calls == []


@pytest.mark.mps
class TestMpsMarks:
    def test_marks_capture_real_gpu_time_by_default(self):
        """Marks sync by default, so they reflect execution, not launch time.

        A non-synced mark over 200 large matmuls would report microseconds
        (kernel-launch only); a synced one reports the real tens+ of ms.
        """
        a = torch.randn(2048, 2048, device="mps")
        with step_perf("mps", batch_size=1) as sp:
            x = torch.randn(2048, 2048, device="mps")
            for _ in range(200):
                x = x @ a
            sp.mark("compute")
        assert sp.perf.marks["compute"] > 0.01

    def test_marks_partition_the_step(self):
        """Because every mark syncs, the marks sum to ~ the whole step time."""
        a = torch.randn(1024, 1024, device="mps")
        with step_perf("mps", batch_size=1) as sp:
            x = torch.randn(1024, 1024, device="mps")
            for _ in range(100):
                x = x @ a
            sp.mark("first")
            for _ in range(100):
                x = x @ a
            sp.mark("second")
        perf = sp.perf
        marks_sum = perf.marks["first"] + perf.marks["second"]
        assert marks_sum <= perf.step_time_sec + 1e-3
        assert marks_sum >= 0.7 * perf.step_time_sec
