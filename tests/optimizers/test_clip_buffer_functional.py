"""Tests for functional clip buffer.

Tests verify that the functional clip buffer:
1. Is purely functional (no side effects, immutable state)
2. Computes correct adaptive clip norms
3. Handles edge cases properly
"""

import pytest
import torch

from opaque.optimizers.adaptive import clip_buffer


class TestClipBufferCreate:
    """Test clip buffer creation."""

    def test_create_basic(self):
        """Test basic buffer creation."""
        state = clip_buffer.create(capacity=100, target_clip_rate=0.20)
        norms_tensor, size = state

        assert isinstance(norms_tensor, torch.Tensor)
        assert len(norms_tensor) == 100
        assert size == 0
        assert torch.all(norms_tensor == 0.0)

    def test_create_validates_capacity(self):
        """Test that create validates capacity."""
        with pytest.raises(ValueError, match="capacity must be positive"):
            clip_buffer.create(capacity=0)

        with pytest.raises(ValueError, match="capacity must be positive"):
            clip_buffer.create(capacity=-1)

    def test_create_validates_target_clip_rate(self):
        """Test that create validates target_clip_rate."""
        with pytest.raises(ValueError, match="target_clip_rate must be in"):
            clip_buffer.create(capacity=100, target_clip_rate=0.0)

        with pytest.raises(ValueError, match="target_clip_rate must be in"):
            clip_buffer.create(capacity=100, target_clip_rate=1.0)

        with pytest.raises(ValueError, match="target_clip_rate must be in"):
            clip_buffer.create(capacity=100, target_clip_rate=1.5)


class TestClipBufferUpdate:
    """Test clip buffer update function."""

    def test_update_immutability(self):
        """Test that update doesn't modify original state."""
        state = clip_buffer.create(capacity=10)
        norms = torch.tensor([0.5, 1.0, 1.5])

        # Update state
        new_state = clip_buffer.update(state, norms)

        # Original state should be unchanged
        old_norms, old_size = state
        assert old_size == 0
        assert torch.all(old_norms == 0.0)

        # New state should have updates
        new_norms, new_size = new_state
        assert new_size == 3
        assert torch.allclose(new_norms[:3], norms)

    def test_update_single_norm(self):
        """Test updating with a single norm."""
        state = clip_buffer.create(capacity=10)
        state = clip_buffer.update(state, torch.tensor([1.5]))

        norms, size = state
        assert size == 1
        assert norms[0] == 1.5

    def test_update_multiple_norms(self):
        """Test updating with multiple norms."""
        state = clip_buffer.create(capacity=10)
        norms = torch.tensor([0.5, 1.0, 1.5, 2.0])
        state = clip_buffer.update(state, norms)

        buffer_norms, size = state
        assert size == 4
        assert torch.allclose(buffer_norms[:4], norms)

    def test_update_with_batch_sizes(self):
        """Test unit normalization with batch sizes."""
        state = clip_buffer.create(capacity=10)

        # Norms: [2.4, 1.8, 3.0]
        # Batch sizes: [2, 2, 3]
        # Unit norms: [1.2, 0.9, 1.0]
        norms = torch.tensor([2.4, 1.8, 3.0])
        batch_sizes = torch.tensor([2.0, 2.0, 3.0])
        state = clip_buffer.update(state, norms, batch_sizes)

        buffer_norms, size = state
        assert size == 3
        expected_unit_norms = torch.tensor([1.2, 0.9, 1.0])
        assert torch.allclose(buffer_norms[:3], expected_unit_norms)

    def test_update_ring_buffer_wrapping(self):
        """Test that buffer wraps around when full."""
        capacity = 5
        state = clip_buffer.create(capacity=capacity)

        # Add 8 norms to a buffer of capacity 5
        norms = torch.arange(1.0, 9.0)  # [1, 2, 3, 4, 5, 6, 7, 8]
        state = clip_buffer.update(state, norms)

        buffer_norms, size = state
        assert size == 8  # Total norms added

        # Buffer should contain the last 5: [6, 7, 8] wrapping to [4, 5, 6, 7, 8]
        # Ring buffer: indices 0,1,2 = [6,7,8], then 3,4 = [4,5]
        # Wait, let me think about this more carefully...
        # We add 1,2,3,4,5,6,7,8 in order
        # Capacity is 5
        # After adding all 8:
        # idx 0 (size=0) = 1, idx 1 (size=1) = 2, idx 2 (size=2) = 3, idx 3 (size=3) = 4, idx 4 (size=4) = 5
        # idx 0 (size=5) = 6, idx 1 (size=6) = 7, idx 2 (size=7) = 8
        # So buffer = [6, 7, 8, 4, 5]
        expected = torch.tensor([6.0, 7.0, 8.0, 4.0, 5.0])
        assert torch.allclose(buffer_norms, expected)


class TestGetAdaptiveClipNorm:
    """Test adaptive clip norm computation."""

    def test_empty_buffer_returns_max(self):
        """Test that empty buffer returns clip_norm_max."""
        state = clip_buffer.create(capacity=10)
        clip_norm = clip_buffer.get_adaptive_clip_norm(
            state, target_clip_rate=0.20, clip_norm_max=5.0
        )
        assert clip_norm == 5.0

    def test_adaptive_clip_norm_percentile(self):
        """Test that adaptive clip norm uses correct percentile."""
        state = clip_buffer.create(capacity=100)

        # Add norms: [0.1, 0.2, ..., 1.0] (10 values)
        norms = torch.linspace(0.1, 1.0, 10)
        state = clip_buffer.update(state, norms)

        # Target clip rate 0.20 → 80th percentile
        # For 10 values, 80th percentile should be around 0.82
        clip_norm = clip_buffer.get_adaptive_clip_norm(state, target_clip_rate=0.20)
        assert 0.7 < clip_norm < 0.9  # Approximate check

    def test_clip_norm_clamping(self):
        """Test that clip norms are clamped to min/max."""
        state = clip_buffer.create(capacity=10)

        # Very small norms
        norms = torch.tensor([0.01, 0.02, 0.03])
        state = clip_buffer.update(state, norms)

        # Should clamp to min
        clip_norm = clip_buffer.get_adaptive_clip_norm(
            state, clip_norm_min=0.5, clip_norm_max=10.0
        )
        assert clip_norm == 0.5

        # Very large norms
        state = clip_buffer.create(capacity=10)
        norms = torch.tensor([50.0, 60.0, 70.0])
        state = clip_buffer.update(state, norms)

        # Should clamp to max
        clip_norm = clip_buffer.get_adaptive_clip_norm(
            state, clip_norm_min=0.1, clip_norm_max=10.0
        )
        assert clip_norm == 10.0


class TestGetClipRate:
    """Test clip rate computation."""

    def test_empty_buffer_returns_zero(self):
        """Test that empty buffer returns 0.0."""
        state = clip_buffer.create(capacity=10)
        rate = clip_buffer.get_clip_rate(state, threshold=1.0)
        assert rate == 0.0

    def test_clip_rate_calculation(self):
        """Test clip rate calculation."""
        state = clip_buffer.create(capacity=10)

        # Norms: [0.5, 1.2, 0.8, 1.5, 0.9]
        # Threshold: 1.0
        # Expected: 2 out of 5 exceed → 40%
        norms = torch.tensor([0.5, 1.2, 0.8, 1.5, 0.9])
        state = clip_buffer.update(state, norms)

        rate = clip_buffer.get_clip_rate(state, threshold=1.0)
        assert rate == pytest.approx(0.4)

    def test_clip_rate_all_below(self):
        """Test clip rate when all norms below threshold."""
        state = clip_buffer.create(capacity=10)
        norms = torch.tensor([0.1, 0.2, 0.3])
        state = clip_buffer.update(state, norms)

        rate = clip_buffer.get_clip_rate(state, threshold=1.0)
        assert rate == 0.0

    def test_clip_rate_all_above(self):
        """Test clip rate when all norms above threshold."""
        state = clip_buffer.create(capacity=10)
        norms = torch.tensor([1.5, 2.0, 2.5])
        state = clip_buffer.update(state, norms)

        rate = clip_buffer.get_clip_rate(state, threshold=1.0)
        assert rate == 1.0


class TestGetSize:
    """Test size getter."""

    def test_initial_size_is_zero(self):
        """Test that initial size is 0."""
        state = clip_buffer.create(capacity=10)
        assert clip_buffer.get_size(state) == 0

    def test_size_increases_with_updates(self):
        """Test that size increases with each update."""
        state = clip_buffer.create(capacity=10)

        state = clip_buffer.update(state, torch.tensor([1.0]))
        assert clip_buffer.get_size(state) == 1

        state = clip_buffer.update(state, torch.tensor([2.0, 3.0]))
        assert clip_buffer.get_size(state) == 3

    def test_size_exceeds_capacity(self):
        """Test that size can exceed capacity (counts total additions)."""
        state = clip_buffer.create(capacity=5)
        norms = torch.arange(1.0, 11.0)  # 10 norms
        state = clip_buffer.update(state, norms)

        # Size should be 10 even though capacity is 5
        assert clip_buffer.get_size(state) == 10


class TestFunctionalProperties:
    """Test functional programming properties."""

    def test_pure_functions(self):
        """Test that all functions are pure (no side effects)."""
        state1 = clip_buffer.create(capacity=10)
        norms = torch.tensor([1.0, 2.0, 3.0])

        # Call update multiple times on same state
        state2a = clip_buffer.update(state1, norms)
        state2b = clip_buffer.update(state1, norms)

        # Should produce identical results
        assert clip_buffer.get_size(state2a) == clip_buffer.get_size(state2b)

        norms_a, size_a = state2a
        norms_b, size_b = state2b
        assert torch.allclose(norms_a, norms_b)
        assert size_a == size_b

        # Original state should be unchanged
        assert clip_buffer.get_size(state1) == 0

    def test_deterministic_operations(self):
        """Test that operations are deterministic."""
        state = clip_buffer.create(capacity=100)
        norms = torch.randn(50)

        # Run same operations twice
        state1 = clip_buffer.update(state, norms)
        clip_norm1 = clip_buffer.get_adaptive_clip_norm(state1)
        clip_rate1 = clip_buffer.get_clip_rate(state1, threshold=1.0)

        state2 = clip_buffer.update(state, norms)
        clip_norm2 = clip_buffer.get_adaptive_clip_norm(state2)
        clip_rate2 = clip_buffer.get_clip_rate(state2, threshold=1.0)

        # Should be identical
        assert clip_norm1 == clip_norm2
        assert clip_rate1 == clip_rate2


class TestIntegration:
    """Integration tests simulating real usage."""

    def test_adaptive_clipping_workflow(self):
        """Test complete adaptive clipping workflow."""
        # Initialize
        capacity = 100
        target_clip_rate = 0.20
        state = clip_buffer.create(capacity=capacity, target_clip_rate=target_clip_rate)

        # Simulate training for 10 steps
        for step in range(10):
            # Simulate per-example gradient norms
            norms = torch.randn(32).abs() + 0.5  # 32 examples, positive norms

            # Update buffer
            state = clip_buffer.update(state, norms)

            # Get adaptive threshold
            clip_norm = clip_buffer.get_adaptive_clip_norm(state, target_clip_rate=target_clip_rate)

            # Check clip rate
            clip_rate = clip_buffer.get_clip_rate(state, threshold=clip_norm)

            # Adaptive threshold should stabilize around target rate after enough samples
            if step > 5:
                # After 6 steps = 192 samples, should be close to target
                assert 0.05 < clip_rate < 0.40  # Roughly in range

    def test_buffer_convergence(self):
        """Test that buffer converges to stable clip norm."""
        state = clip_buffer.create(capacity=1000, target_clip_rate=0.20)

        # Add many norms from same distribution
        clip_norms = []
        for _ in range(100):
            norms = torch.randn(10).abs() + 1.0  # Mean around 1.0
            state = clip_buffer.update(state, norms)
            clip_norm = clip_buffer.get_adaptive_clip_norm(state)
            clip_norms.append(clip_norm)

        # Clip norm should stabilize (last 10 values should be similar)
        recent_clip_norms = clip_norms[-10:]
        std = torch.tensor(recent_clip_norms).std()
        assert std < 0.2  # Should be fairly stable


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
