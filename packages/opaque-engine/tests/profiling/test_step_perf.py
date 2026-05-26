"""Tests for the redesigned profiling API (StepPerf, step_perf, PerfState)."""

import time

import pytest
import torch

from opaque.profiling import PerfState, StepPerf, step_perf


DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.insert(0, "cuda")
if torch.backends.mps.is_available():
    DEVICES.insert(0, "mps")


class TestStepPerf:
    """Tests for StepPerf frozen dataclass."""

    def test_default_values(self):
        perf = StepPerf()
        assert perf.step_time_sec == 0.0
        assert perf.samples_per_second == 0.0
        assert perf.steps_per_second == 0.0
        assert perf.memory_peak_gb == 0.0
        assert perf.batch_size == 0
        assert perf.marks == {}

    def test_to_dict_bare_keys(self):
        perf = StepPerf(step_time_sec=1.5, samples_per_second=20.0, memory_peak_gb=3.0)
        d = perf.to_dict()
        assert "step_time_sec" in d
        assert "samples_per_second" in d
        assert "memory_peak_gb" in d
        assert d["step_time_sec"] == 1.5
        assert d["samples_per_second"] == 20.0

    def test_to_dict_with_prefix(self):
        perf = StepPerf(step_time_sec=1.0)
        d = perf.to_dict(prefix="train/")
        assert "train/step_time_sec" in d
        assert "train/samples_per_second" in d
        assert "step_time_sec" not in d

    def test_to_dict_includes_marks(self):
        perf = StepPerf(step_time_sec=2.0, marks={"clip": 1.2, "noise": 0.5})
        d = perf.to_dict(prefix="perf/")
        assert d["perf/clip_sec"] == 1.2
        assert d["perf/noise_sec"] == 0.5

    def test_frozen(self):
        perf = StepPerf(step_time_sec=1.0)
        with pytest.raises(AttributeError):
            perf.step_time_sec = 2.0


class TestStepPerfContextManager:
    """Tests for step_perf() context manager."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_measures_time(self, device):
        with step_perf(device, batch_size=32) as sp:
            x = torch.randn(100, 100)
            _ = x @ x.T
        result = sp.result
        assert result.step_time_sec > 0
        assert result.batch_size == 32
        assert result.samples_per_second > 0
        assert result.steps_per_second > 0

    @pytest.mark.parametrize("device", DEVICES)
    def test_result_not_available_inside_context(self, device):
        with step_perf(device) as sp:
            with pytest.raises(RuntimeError, match="not available"):
                _ = sp.result

    def test_marks_basic(self):
        with step_perf("cpu", batch_size=10) as sp:
            time.sleep(0.01)
            sp.mark("phase_a")
            time.sleep(0.01)
            sp.mark("phase_b")
        result = sp.result
        assert "phase_a" in result.marks
        assert "phase_b" in result.marks
        assert result.marks["phase_a"] > 0
        assert result.marks["phase_b"] > 0

    def test_marks_appear_in_to_dict(self):
        with step_perf("cpu") as sp:
            sp.mark("clip")
            sp.mark("noise")
        d = sp.result.to_dict(prefix="train/")
        assert "train/clip_sec" in d
        assert "train/noise_sec" in d

    def test_no_memory_tracking(self):
        with step_perf("cpu", track_memory=False) as sp:
            pass
        result = sp.result
        assert result.memory_peak_gb == 0.0
        assert result.memory_allocated_gb == 0.0

    def test_string_device(self):
        with step_perf("cpu", batch_size=16) as sp:
            pass
        assert sp.result.batch_size == 16
        assert sp.result.step_time_sec > 0

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    def test_mps_synchronizes(self, monkeypatch):
        calls = {"count": 0}

        def _sync():
            calls["count"] += 1

        monkeypatch.setattr(torch.mps, "synchronize", _sync)

        with step_perf("mps") as _sp:
            pass
        # sync called twice: once before timing, once after
        assert calls["count"] == 2


class TestPerfState:
    """Tests for PerfState functional accumulator."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_initialization(self, device):
        state = PerfState(device=torch.device(device))
        assert state.num_steps == 0
        assert state.total_time == 0.0
        assert state.total_samples == 0
        assert state.max_peak_memory_gb == 0.0
        assert state.last_step is None

    def test_add_single_step(self):
        state = PerfState(device=torch.device("cpu"))
        perf = StepPerf(step_time_sec=1.0, batch_size=32, memory_peak_gb=2.0)
        state = state.add(perf)
        assert state.num_steps == 1
        assert state.total_time == 1.0
        assert state.total_samples == 32
        assert state.max_peak_memory_gb == 2.0
        assert state.last_step is perf

    def test_warmup_exclusion(self):
        state = PerfState(device=torch.device("cpu"))
        # First step (warmup) — excluded from stable averages
        warmup = StepPerf(step_time_sec=10.0, batch_size=32)
        state = state.add(warmup)
        assert state.num_steps_stable == 0
        assert state.avg_step_time_stable == 10.0  # falls back to avg_step_time

        # Second step — included in stable averages
        step2 = StepPerf(step_time_sec=2.0, batch_size=32)
        state = state.add(step2)
        assert state.num_steps_stable == 1
        assert state.avg_step_time_stable == pytest.approx(2.0)

    def test_avg_step_time_stable_multi(self):
        state = PerfState(device=torch.device("cpu"))
        state = state.add(StepPerf(step_time_sec=10.0, batch_size=1))  # warmup
        state = state.add(StepPerf(step_time_sec=2.0, batch_size=1))
        state = state.add(StepPerf(step_time_sec=2.0, batch_size=1))
        state = state.add(StepPerf(step_time_sec=2.0, batch_size=1))
        assert state.avg_step_time_stable == pytest.approx(2.0)

    def test_avg_samples_per_second_stable(self):
        state = PerfState(device=torch.device("cpu"))
        state = state.add(StepPerf(step_time_sec=10.0, batch_size=32))  # warmup
        state = state.add(StepPerf(step_time_sec=1.0, batch_size=32))
        state = state.add(StepPerf(step_time_sec=1.0, batch_size=32))
        assert state.avg_samples_per_second_stable == pytest.approx(32.0)

    def test_max_peak_memory_tracking(self):
        state = PerfState(device=torch.device("cpu"))
        state = state.add(StepPerf(memory_peak_gb=2.0))
        state = state.add(StepPerf(memory_peak_gb=5.0))
        state = state.add(StepPerf(memory_peak_gb=3.0))
        assert state.max_peak_memory_gb == 5.0

    def test_to_dict(self):
        state = PerfState(device=torch.device("cpu"))
        state = state.add(StepPerf(step_time_sec=1.0, batch_size=32, memory_peak_gb=4.0))
        d = state.to_dict(prefix="perf/")
        assert "perf/avg_step_time_sec" in d
        assert "perf/avg_samples_per_second" in d
        assert "perf/max_peak_memory_gb" in d
        assert d["perf/max_peak_memory_gb"] == 4.0

    def test_to_dict_bare_keys(self):
        state = PerfState(device=torch.device("cpu"))
        state = state.add(StepPerf(step_time_sec=1.0))
        d = state.to_dict()
        assert "avg_step_time_sec" in d

    def test_immutable(self):
        state = PerfState(device=torch.device("cpu"))
        new_state = state.add(StepPerf(step_time_sec=1.0))
        assert state.num_steps == 0  # original unchanged
        assert new_state.num_steps == 1

    def test_empty_state_properties(self):
        state = PerfState(device=torch.device("cpu"))
        assert state.avg_step_time == 0.0
        assert state.avg_step_time_stable == 0.0
        assert state.avg_samples_per_second == 0.0
        assert state.avg_samples_per_second_stable == 0.0


class TestStepPerfIntegration:
    """Integration test: step_perf + PerfState together."""

    def test_full_loop(self):
        state = PerfState(device=torch.device("cpu"))
        for i in range(3):
            with step_perf("cpu", batch_size=16) as sp:
                x = torch.randn(50, 50)
                _ = x @ x.T
                sp.mark("compute")
            state = state.add(sp.result)

        assert state.num_steps == 3
        assert state.total_samples == 48
        assert state.avg_step_time > 0
        assert state.last_step is not None
        assert "compute" in state.last_step.marks
