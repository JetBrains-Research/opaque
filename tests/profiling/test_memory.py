"""Tests for memory profiling tools.

Tests the actual profiling API (profile_memory, find_max_microbatch_size)
across different devices (CUDA, MPS, CPU).
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from opaque.profiling import (
    MemoryProfile,
    MemoryProfiler,
    MemoryTracker,
    find_max_microbatch_size,
    profile_memory,
)
from opaque.utils import make_functional

# Test on all available devices
DEVICES = []
if torch.cuda.is_available():
    DEVICES.append("cuda")
if torch.backends.mps.is_available():
    DEVICES.append("mps")
DEVICES.append("cpu")  # Always test CPU (with warnings)


# Simple test models
class TinyModel(nn.Module):
    """Tiny model for fast testing (~40KB)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 10)

    def forward(self, x):
        return self.fc(x)


class SmallModel(nn.Module):
    """Small model for testing (~160KB)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def create_sample_batch(batch_size: int, device: str = "cpu"):
    """Create sample batch for testing."""
    data = torch.randn(batch_size, 100, device=device)
    targets = torch.randint(0, 10, (batch_size,), device=device)
    return data, targets


def create_loss_fn(model: nn.Module):
    """Create loss function for testing."""

    def loss_fn(params, data, targets):
        # Need to reconstruct functional call
        fmodel, trainable, frozen = make_functional(model, partition_trainable=True)
        all_params = {**frozen, **trainable}
        output = fmodel(all_params, data)
        return F.cross_entropy(output, targets, reduction="sum")

    return loss_fn


class TestMemoryTracker:
    """Tests for MemoryTracker class."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_initialization(self, device):
        """Should initialize tracker for device."""
        tracker = MemoryTracker(device)
        assert tracker.device == device

        if device in ["cuda", "mps"]:
            # Check if actually available
            if device == "cuda":
                assert tracker.is_supported() == torch.cuda.is_available()
            elif device == "mps":
                assert tracker.is_supported() == torch.backends.mps.is_available()
        else:
            assert not tracker.is_supported()

    def test_cuda_support(self):
        """Should detect CUDA support correctly."""
        tracker = MemoryTracker("cuda")
        assert tracker.is_supported() == torch.cuda.is_available()

    def test_mps_support(self):
        """Should detect MPS support correctly."""
        tracker = MemoryTracker("mps")
        assert tracker.is_supported() == torch.backends.mps.is_available()

    def test_cpu_not_supported(self):
        """Should report CPU as not supported."""
        tracker = MemoryTracker("cpu")
        assert not tracker.is_supported()


@pytest.mark.parametrize("device", DEVICES)
class TestProfileMemory:
    """Tests for profile_memory function."""

    def test_profiles_full_batch(self, device):
        """Should profile memory for full batch."""
        model = TinyModel().to(device)
        data, targets = create_sample_batch(16, device=device)
        loss_fn = create_loss_fn(model)

        profile = profile_memory(
            model,
            (data, targets),
            loss_fn,
            l2_clip_norm=1.0,
        )

        # Check basic attributes
        assert isinstance(profile, MemoryProfile)
        assert profile.batch_size == 16
        assert profile.microbatch_size is None
        assert profile.device == device

        # Check status based on device support
        if device in ["cuda", "mps"]:
            # Full profiling supported
            assert profile.status in ["ok", "warning", "critical"]
            assert profile.peak_gb >= 0
            assert profile.available_gb > 0
        else:
            # CPU - limited support
            assert profile.status == "unsupported"
            assert profile.peak_gb == 0
            assert profile.available_gb == 0

    def test_profiles_with_microbatch(self, device):
        """Should profile memory with microbatching."""
        model = TinyModel().to(device)
        data, targets = create_sample_batch(16, device=device)
        loss_fn = create_loss_fn(model)

        profile = profile_memory(
            model,
            (data, targets),
            loss_fn,
            l2_clip_norm=1.0,
            microbatch_size=4,
        )

        assert profile.batch_size == 16
        assert profile.microbatch_size == 4
        assert profile.device == device

    def test_utilization_calculation(self, device):
        """Should calculate memory utilization correctly."""
        model = TinyModel().to(device)
        data, targets = create_sample_batch(8, device=device)
        loss_fn = create_loss_fn(model)

        profile = profile_memory(
            model,
            (data, targets),
            loss_fn,
            l2_clip_norm=1.0,
        )

        util = profile.utilization()
        assert 0.0 <= util <= 1.0

        if device in ["cuda", "mps"] and profile.available_gb > 0:
            # Should have meaningful utilization
            expected = profile.peak_gb / profile.available_gb
            assert util == pytest.approx(expected)

    def test_str_representation(self, device):
        """Should have human-readable string representation."""
        model = TinyModel().to(device)
        data, targets = create_sample_batch(8, device=device)
        loss_fn = create_loss_fn(model)

        profile = profile_memory(
            model,
            (data, targets),
            loss_fn,
            l2_clip_norm=1.0,
        )

        profile_str = str(profile)
        assert "Memory Profile" in profile_str
        assert "batch_size=8" in profile_str
        assert device in profile_str.lower()


# Only test find_max_microbatch_size on supported devices
@pytest.mark.parametrize("device", [d for d in DEVICES if d in ["cuda", "mps"]])
class TestFindMaxMicrobatchSize:
    """Tests for find_max_microbatch_size function (CUDA/MPS only)."""

    def test_finds_valid_size(self, device):
        """Should find a valid microbatch size."""
        # Skip if device not actually available
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if device == "mps" and not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        model = TinyModel().to(device)
        data, targets = create_sample_batch(32, device=device)
        loss_fn = create_loss_fn(model)

        max_size = find_max_microbatch_size(
            model,
            (data, targets),
            batch_size=32,
            loss_fn=loss_fn,
            l2_clip_norm=1.0,
        )

        # Should return power of 2
        assert max_size in [1, 2, 4, 8, 16, 32]
        assert max_size >= 1
        assert max_size <= 32

    def test_returns_at_least_min_size(self, device):
        """Should return at least min_size."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if device == "mps" and not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        model = TinyModel().to(device)
        data, targets = create_sample_batch(16, device=device)
        loss_fn = create_loss_fn(model)

        max_size = find_max_microbatch_size(
            model,
            (data, targets),
            batch_size=16,
            loss_fn=loss_fn,
            l2_clip_norm=1.0,
            min_size=2,
        )

        assert max_size >= 2

    def test_respects_safety_margin(self, device):
        """Should respect safety margin parameter."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if device == "mps" and not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        model = TinyModel().to(device)
        data, targets = create_sample_batch(32, device=device)
        loss_fn = create_loss_fn(model)

        # Conservative safety margin
        max_size_conservative = find_max_microbatch_size(
            model,
            (data, targets),
            batch_size=32,
            loss_fn=loss_fn,
            l2_clip_norm=1.0,
            safety_margin=0.5,  # Very conservative
        )

        # Aggressive safety margin
        max_size_aggressive = find_max_microbatch_size(
            model,
            (data, targets),
            batch_size=32,
            loss_fn=loss_fn,
            l2_clip_norm=1.0,
            safety_margin=0.95,  # Aggressive
        )

        # Aggressive should allow equal or larger size
        assert max_size_aggressive >= max_size_conservative


class TestCPUWarnings:
    """Test that CPU profiling produces appropriate warnings."""

    def test_profile_memory_warns_on_cpu(self):
        """Should warn when profiling on CPU."""
        model = TinyModel()
        data, targets = create_sample_batch(8, device="cpu")
        loss_fn = create_loss_fn(model)

        with pytest.warns(UserWarning, match="CPU is limited"):
            profile = profile_memory(
                model,
                (data, targets),
                loss_fn,
                l2_clip_norm=1.0,
            )

        assert profile.status == "unsupported"

    def test_find_max_warns_on_cpu(self):
        """Should warn when finding max size on CPU."""
        model = TinyModel()
        data, targets = create_sample_batch(16, device="cpu")
        loss_fn = create_loss_fn(model)

        with pytest.warns(UserWarning, match="CPU is not supported"):
            max_size = find_max_microbatch_size(
                model,
                (data, targets),
                batch_size=16,
                loss_fn=loss_fn,
                l2_clip_norm=1.0,
            )

        # Should return min_size (default 1)
        assert max_size == 1


class TestIntegration:
    """Integration tests combining profiling with actual training."""

    @pytest.mark.skipif(
        not (torch.cuda.is_available() or torch.backends.mps.is_available()),
        reason="Requires CUDA or MPS",
    )
    def test_profile_then_train(self):
        """Should be able to profile then train with recommended config."""
        # Detect available device
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("No GPU available")

        model = SmallModel().to(device)
        data, targets = create_sample_batch(32, device=device)
        loss_fn = create_loss_fn(model)

        # Step 1: Profile
        profile = profile_memory(
            model,
            (data, targets),
            loss_fn,
            l2_clip_norm=1.0,
        )

        assert profile.status in ["ok", "warning", "critical"]

        # Step 2: Find optimal microbatch size
        max_mb = find_max_microbatch_size(
            model,
            (data, targets),
            batch_size=32,
            loss_fn=loss_fn,
            l2_clip_norm=1.0,
        )

        assert max_mb >= 1

        # Step 3: Train with recommended config
        from opaque import clipped_grad

        grad_fn, clip_state = clipped_grad(
            loss_fn,
            l2_clip_norm=1.0,
            batch_argnums=(1, 2),
            microbatch_size=max_mb,
        )

        # Make functional
        fmodel, trainable, frozen = make_functional(model, partition_trainable=True)
        params = {**frozen, **trainable}

        # Should successfully compute gradients
        grads, _ = grad_fn(params, data, targets, state=clip_state)

        # Check gradients exist
        assert isinstance(grads, dict)
        assert len(grads) > 0


class TestMemoryProfiler:
    """Tests for MemoryProfiler context manager."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_initialization(self, device):
        """Should initialize profiler for device."""
        profiler = MemoryProfiler(device=device)
        assert profiler.device == device
        assert len(profiler.snapshots) == 0
        assert not profiler._active

    def test_auto_detect_device(self):
        """Should auto-detect device when not specified."""
        profiler = MemoryProfiler()
        assert profiler.device in ["cuda", "mps", "cpu"]

    @pytest.mark.parametrize("device", DEVICES)
    def test_context_manager_basic(self, device):
        """Should work as context manager."""
        profiler = MemoryProfiler(device=device)

        with profiler:
            assert profiler._active
            assert len(profiler.snapshots) == 1
            assert profiler.snapshots[0].label == "start"

        # After exit
        assert not profiler._active
        assert len(profiler.snapshots) == 2
        assert profiler.snapshots[-1].label == "end"

    @pytest.mark.parametrize("device", DEVICES)
    def test_mark_inside_context(self, device):
        """Should record marks inside context."""
        profiler = MemoryProfiler(device=device)

        with profiler:
            profiler.mark("checkpoint1")
            profiler.mark("checkpoint2")

        # Should have: start, checkpoint1, checkpoint2, end
        assert len(profiler.snapshots) == 4
        assert profiler.snapshots[0].label == "start"
        assert profiler.snapshots[1].label == "checkpoint1"
        assert profiler.snapshots[2].label == "checkpoint2"
        assert profiler.snapshots[3].label == "end"

    @pytest.mark.parametrize("device", DEVICES)
    def test_mark_outside_context_raises(self, device):
        """Should raise error when marking outside context."""
        profiler = MemoryProfiler(device=device)

        with pytest.raises(RuntimeError, match="can only be called within"):
            profiler.mark("invalid")

    @pytest.mark.parametrize("device", DEVICES)
    def test_delta_calculation(self, device):
        """Should calculate memory deltas correctly."""
        if device == "cpu":
            pytest.skip("CPU profiling not fully supported")

        profiler = MemoryProfiler(device=device)

        with profiler:
            # Create some tensors to change memory
            _x = torch.randn(1000, 1000, device=device)  # noqa: F841
            profiler.mark("after_tensor")
            _y = torch.randn(1000, 1000, device=device)  # noqa: F841
            profiler.mark("after_second_tensor")

        # Check that we have memory snapshots
        assert len(profiler.snapshots) == 4

        # First snapshot should have 0 delta
        assert profiler.snapshots[0].delta_gb == 0.0

        # Later snapshots should have deltas (could be positive or negative)
        for snapshot in profiler.snapshots[1:]:
            assert isinstance(snapshot.delta_gb, float)

    @pytest.mark.parametrize("device", DEVICES)
    def test_get_peak_memory(self, device):
        """Should report peak memory."""
        profiler = MemoryProfiler(device=device)

        with profiler:
            # Allocate some tensors
            _x = torch.randn(100, 100, device=device)  # noqa: F841
            profiler.mark("allocated")

        peak = profiler.get_peak_memory()
        assert isinstance(peak, float)
        assert peak >= 0.0

    @pytest.mark.parametrize("device", DEVICES)
    def test_get_total_memory(self, device):
        """Should report total memory."""
        profiler = MemoryProfiler(device=device)

        with profiler:
            pass

        total = profiler.get_total_memory()
        assert isinstance(total, float)
        assert total >= 0.0

        # On supported devices, should have meaningful total
        if device in ["cuda", "mps"]:
            if (device == "cuda" and torch.cuda.is_available()) or (
                device == "mps" and torch.backends.mps.is_available()
            ):
                assert total > 0.0

    @pytest.mark.parametrize("device", DEVICES)
    def test_report_format(self, device):
        """Should generate formatted report."""
        profiler = MemoryProfiler(device=device)

        with profiler:
            profiler.mark("step1")
            profiler.mark("step2")

        report = profiler.report()

        # Check report contains expected elements
        assert "Memory Profile Report" in report
        assert device.upper() in report
        assert "Peak Memory" in report
        assert "Total Available" in report
        assert "Timeline" in report
        assert "start" in report
        assert "step1" in report
        assert "step2" in report
        assert "end" in report

    @pytest.mark.parametrize("device", DEVICES)
    def test_empty_report(self, device):
        """Should handle empty profiler gracefully."""
        profiler = MemoryProfiler(device=device)
        report = profiler.report()
        assert "No profiling data" in report

    @pytest.mark.skipif(
        not (torch.cuda.is_available() or torch.backends.mps.is_available()),
        reason="Requires CUDA or MPS",
    )
    def test_profile_training_step(self):
        """Should profile complete training step."""
        # Detect available device
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("No GPU available")

        model = SmallModel().to(device)
        data, targets = create_sample_batch(32, device=device)
        loss_fn = create_loss_fn(model)

        # Make functional
        from opaque import clipped_grad

        fmodel, trainable, frozen = make_functional(model, partition_trainable=True)
        params = {**frozen, **trainable}

        grad_fn, clip_state = clipped_grad(
            loss_fn,
            l2_clip_norm=1.0,
            batch_argnums=(1, 2),
            microbatch_size=8,
        )

        # Profile the training step
        profiler = MemoryProfiler(device=device)

        with profiler:
            grads, _ = grad_fn(params, data, targets, state=clip_state)
            profiler.mark("after_grad")

            # Simulate optimizer step
            for key in grads:
                params[key] = params[key] - 0.01 * grads[key]
            profiler.mark("after_optimizer")

        # Check we captured all stages
        assert len(profiler.snapshots) >= 4
        assert "after_grad" in [s.label for s in profiler.snapshots]
        assert "after_optimizer" in [s.label for s in profiler.snapshots]

        # Report should be valid
        report = profiler.report()
        assert "after_grad" in report
        assert "after_optimizer" in report

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Requires CUDA for detailed stats",
    )
    def test_detailed_report_cuda(self):
        """Should generate detailed report with CUDA stats."""
        device = "cuda"
        model = SmallModel().to(device)
        data, targets = create_sample_batch(32, device=device)
        loss_fn = create_loss_fn(model)

        # Make functional
        from opaque import clipped_grad

        fmodel, trainable, frozen = make_functional(model, partition_trainable=True)
        params = {**frozen, **trainable}

        grad_fn, clip_state = clipped_grad(
            loss_fn,
            l2_clip_norm=1.0,
            batch_argnums=(1, 2),
            microbatch_size=8,
        )

        # Profile with detailed stats
        profiler = MemoryProfiler(device=device)

        with profiler:
            grads, _ = grad_fn(params, data, targets, state=clip_state)
            profiler.mark("after_grad")

        # Get both standard and detailed reports
        standard_report = profiler.report(detailed=False)
        detailed_report = profiler.report(detailed=True)

        # Standard report should have basic info
        assert "Memory Profile Report" in standard_report
        assert "after_grad" in standard_report

        # Detailed report should have CUDA-specific info
        assert "Memory Profile Report" in detailed_report
        assert "Reserved" in detailed_report or "Efficiency" in detailed_report
        assert "CUDA Memory Stats" in detailed_report

        # Check that snapshots have CUDA stats
        for snapshot in profiler.snapshots:
            assert snapshot.reserved_gb is not None
            assert snapshot.reserved_gb >= snapshot.allocated_gb

    def test_component_tracking(self):
        """Should track tensor components when requested."""
        device = "cpu"  # Use CPU for deterministic test
        model = SmallModel().to(device)

        # Create a simple training scenario
        model.train()
        x = torch.randn(4, 100)
        y = torch.randint(0, 10, (4,))

        profiler = MemoryProfiler(device=device)

        with profiler:
            # Mark with component tracking
            profiler.mark("initial", track_components=True)

            # Forward pass
            output = model(x)
            loss = torch.nn.functional.cross_entropy(output, y)

            profiler.mark("after_forward", track_components=True)

            # Backward pass
            loss.backward()

            profiler.mark("after_backward", track_components=True)

        # Check that components were tracked
        for snapshot in profiler.snapshots:
            if snapshot.label in ["initial", "after_forward", "after_backward"]:
                assert snapshot.components is not None
                assert "parameters" in snapshot.components
                assert "gradients" in snapshot.components
                assert "activations" in snapshot.components
                assert "other" in snapshot.components

        # Report should include component breakdown
        report = profiler.report()
        assert "Component Breakdown" in report
        assert "Params" in report
        assert "Grads" in report
        assert "Activations" in report
