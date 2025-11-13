"""Tests for ClipNormBuffer."""

import pytest
import torch

from opaque.adaptive import ClipNormBuffer


class TestClipNormBufferInit:
    """Tests for ClipNormBuffer initialization."""

    def test_default_init(self):
        """Test default initialization."""
        buffer = ClipNormBuffer()
        assert buffer.capacity == 1000
        assert buffer.target_clip_rate == 0.20
        assert len(buffer) == 0

    def test_custom_capacity(self):
        """Test custom capacity."""
        buffer = ClipNormBuffer(capacity=500)
        assert buffer.capacity == 500

    def test_custom_target_clip_rate(self):
        """Test custom target clip rate."""
        buffer = ClipNormBuffer(target_clip_rate=0.30)
        assert buffer.target_clip_rate == 0.30

    def test_invalid_capacity(self):
        """Test that invalid capacity raises error."""
        with pytest.raises(ValueError, match="capacity must be positive"):
            ClipNormBuffer(capacity=0)

        with pytest.raises(ValueError, match="capacity must be positive"):
            ClipNormBuffer(capacity=-10)

    def test_invalid_target_clip_rate(self):
        """Test that invalid target_clip_rate raises error."""
        with pytest.raises(ValueError, match="target_clip_rate must be in"):
            ClipNormBuffer(target_clip_rate=0.0)

        with pytest.raises(ValueError, match="target_clip_rate must be in"):
            ClipNormBuffer(target_clip_rate=1.0)

        with pytest.raises(ValueError, match="target_clip_rate must be in"):
            ClipNormBuffer(target_clip_rate=1.5)


class TestClipNormBufferUpdate:
    """Tests for buffer updates."""

    def test_update_single_norm(self):
        """Test updating with single gradient norm."""
        buffer = ClipNormBuffer(capacity=10)
        buffer.update(torch.tensor([1.5]))
        assert len(buffer) == 1

    def test_update_multiple_norms(self):
        """Test updating with multiple norms."""
        buffer = ClipNormBuffer(capacity=10)
        norms = torch.tensor([1.0, 1.5, 2.0, 0.8, 1.2])
        buffer.update(norms)
        assert len(buffer) == 5

    def test_update_with_batch_sizes(self):
        """Test unit normalization with batch sizes."""
        buffer = ClipNormBuffer(capacity=10)
        # Norms before normalization
        norms = torch.tensor([2.0, 3.0, 1.5])
        batch_sizes = torch.tensor([2.0, 3.0, 1.0])
        buffer.update(norms, batch_sizes)

        # After update, buffer should contain: [1.0, 1.0, 1.5]
        assert len(buffer) == 3

    def test_capacity_limit(self):
        """Test that buffer respects capacity limit."""
        buffer = ClipNormBuffer(capacity=5)

        # Add 10 norms (should keep only last 5)
        for i in range(10):
            buffer.update(torch.tensor([float(i)]))

        assert len(buffer) == 5
        # Should contain norms from steps 5-9
        assert list(buffer.buffer) == [5.0, 6.0, 7.0, 8.0, 9.0]

    def test_update_scalar(self):
        """Test updating with scalar tensors."""
        buffer = ClipNormBuffer(capacity=10)
        buffer.update(torch.tensor(1.5))
        assert len(buffer) == 1

    def test_update_batched(self):
        """Test multiple updates accumulate."""
        buffer = ClipNormBuffer(capacity=20)
        buffer.update(torch.tensor([1.0, 2.0]))
        buffer.update(torch.tensor([3.0, 4.0]))
        assert len(buffer) == 4


class TestClipNormBufferAdaptiveThreshold:
    """Tests for adaptive clip norm computation."""

    def test_empty_buffer_returns_default(self):
        """Test that empty buffer returns default value."""
        buffer = ClipNormBuffer()
        clip_norm = buffer.get_adaptive_clip_norm()
        assert clip_norm == 1.0

    def test_percentile_computation(self):
        """Test percentile-based threshold computation."""
        buffer = ClipNormBuffer(target_clip_rate=0.20)  # 80th percentile

        # Add 10 norms: [1, 2, 3, ..., 10]
        norms = torch.arange(1.0, 11.0)
        buffer.update(norms)

        # 80th percentile of [1..10] should be 8.2
        clip_norm = buffer.get_adaptive_clip_norm()
        assert 8.0 <= clip_norm <= 9.0  # Allow some tolerance

    def test_clamping_to_min(self):
        """Test that clip norm is clamped to minimum."""
        buffer = ClipNormBuffer(target_clip_rate=0.20)
        buffer.update(torch.tensor([0.01, 0.02, 0.03]))

        clip_norm = buffer.get_adaptive_clip_norm(clip_norm_min=0.5)
        assert clip_norm >= 0.5

    def test_clamping_to_max(self):
        """Test that clip norm is clamped to maximum."""
        buffer = ClipNormBuffer(target_clip_rate=0.20)
        buffer.update(torch.tensor([100.0, 200.0, 300.0]))

        clip_norm = buffer.get_adaptive_clip_norm(clip_norm_max=50.0)
        assert clip_norm <= 50.0

    def test_different_target_rates(self):
        """Test different target clip rates give different percentiles."""
        norms = torch.arange(1.0, 101.0)  # 1 to 100

        # Target 10% clipping → 90th percentile
        buffer_10 = ClipNormBuffer(target_clip_rate=0.10)
        buffer_10.update(norms)
        clip_norm_10 = buffer_10.get_adaptive_clip_norm(clip_norm_max=200.0)

        # Target 50% clipping → 50th percentile (median)
        buffer_50 = ClipNormBuffer(target_clip_rate=0.50)
        buffer_50.update(norms)
        clip_norm_50 = buffer_50.get_adaptive_clip_norm(clip_norm_max=200.0)

        # Higher percentile should give larger clip norm
        assert clip_norm_10 > clip_norm_50


class TestClipNormBufferClipRate:
    """Tests for clip rate computation."""

    def test_empty_buffer_clip_rate(self):
        """Test clip rate of empty buffer."""
        buffer = ClipNormBuffer()
        clip_rate = buffer.get_clip_rate(1.0)
        assert clip_rate == 0.0

    def test_all_below_threshold(self):
        """Test when all norms below threshold."""
        buffer = ClipNormBuffer()
        buffer.update(torch.tensor([0.5, 0.8, 0.9, 0.7]))

        clip_rate = buffer.get_clip_rate(1.0)
        assert clip_rate == 0.0

    def test_all_above_threshold(self):
        """Test when all norms above threshold."""
        buffer = ClipNormBuffer()
        buffer.update(torch.tensor([1.5, 2.0, 3.0, 1.8]))

        clip_rate = buffer.get_clip_rate(1.0)
        assert clip_rate == 1.0

    def test_partial_clipping(self):
        """Test when some norms exceed threshold."""
        buffer = ClipNormBuffer()
        # 3 out of 5 exceed threshold of 1.0
        buffer.update(torch.tensor([0.5, 1.2, 0.8, 1.5, 2.0]))

        clip_rate = buffer.get_clip_rate(1.0)
        assert clip_rate == pytest.approx(0.6, abs=0.01)  # 3/5 = 0.6

    def test_clip_rate_changes_with_threshold(self):
        """Test that clip rate depends on threshold."""
        buffer = ClipNormBuffer()
        buffer.update(torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5]))

        # Low threshold → high clip rate
        rate_low = buffer.get_clip_rate(0.7)
        # High threshold → low clip rate
        rate_high = buffer.get_clip_rate(2.2)

        assert rate_low > rate_high


class TestClipNormBufferUtilities:
    """Tests for utility methods."""

    def test_len(self):
        """Test __len__ method."""
        buffer = ClipNormBuffer()
        assert len(buffer) == 0

        buffer.update(torch.tensor([1.0, 2.0, 3.0]))
        assert len(buffer) == 3

    def test_clear(self):
        """Test clear method."""
        buffer = ClipNormBuffer()
        buffer.update(torch.tensor([1.0, 2.0, 3.0]))
        assert len(buffer) == 3

        buffer.clear()
        assert len(buffer) == 0

    def test_clear_resets_statistics(self):
        """Test that clear resets all statistics."""
        buffer = ClipNormBuffer()
        buffer.update(torch.tensor([1.0, 2.0, 3.0]))

        buffer.clear()

        # After clear, should return defaults
        assert buffer.get_adaptive_clip_norm() == 1.0
        assert buffer.get_clip_rate(1.0) == 0.0


class TestClipNormBufferEdgeCases:
    """Tests for edge cases and corner scenarios."""

    def test_single_norm(self):
        """Test with only one norm in buffer."""
        buffer = ClipNormBuffer(target_clip_rate=0.20)
        buffer.update(torch.tensor([5.0]))

        # With one norm, percentile should return that norm
        clip_norm = buffer.get_adaptive_clip_norm()
        assert clip_norm == pytest.approx(5.0, abs=0.1)

    def test_identical_norms(self):
        """Test with all identical norms."""
        buffer = ClipNormBuffer(target_clip_rate=0.20)
        buffer.update(torch.ones(100) * 3.0)

        # All percentiles should be 3.0
        clip_norm = buffer.get_adaptive_clip_norm()
        assert clip_norm == pytest.approx(3.0, abs=0.1)

    def test_very_large_norms(self):
        """Test with very large gradient norms."""
        buffer = ClipNormBuffer()
        buffer.update(torch.tensor([1e6, 1e7, 1e8]))

        # Should be clamped to max
        clip_norm = buffer.get_adaptive_clip_norm(clip_norm_max=100.0)
        assert clip_norm == 100.0

    def test_very_small_norms(self):
        """Test with very small gradient norms."""
        buffer = ClipNormBuffer()
        buffer.update(torch.tensor([1e-6, 1e-7, 1e-8]))

        # Should be clamped to min
        clip_norm = buffer.get_adaptive_clip_norm(clip_norm_min=0.1)
        assert clip_norm == 0.1

    def test_mixed_batch_sizes(self):
        """Test with varying batch sizes for microbatching."""
        buffer = ClipNormBuffer()

        # Different batch sizes
        norms = torch.tensor([4.0, 6.0, 3.0])  # Raw norms
        sizes = torch.tensor([2.0, 3.0, 1.0])  # Batch sizes

        buffer.update(norms, sizes)

        # Unit norms should be [2.0, 2.0, 3.0]
        # With target_clip_rate=0.20 (80th percentile), should be around 2.6
        clip_norm = buffer.get_adaptive_clip_norm()
        assert 2.0 <= clip_norm <= 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
