"""Tests for clip-rate-based learning rate adjustment."""

import pytest

from opaque.adaptive import clip_rate_based_lr_adjustment, compute_clip_rate_thresholds


class TestClipRateBasedLRAdjustment:
    """Tests for clip rate based LR adjustment."""

    def test_clip_rate_in_range_no_change(self):
        """Test that LR doesn't change when clip rate is in acceptable range."""
        gamma = 1.0
        new_gamma = clip_rate_based_lr_adjustment(
            current_lr_multiplier=gamma,
            clip_rate=0.20,  # In target range
            target_clip_rate=0.20,
            clip_rate_low=0.10,
            clip_rate_high=0.30,
        )
        assert new_gamma == pytest.approx(1.0)

    def test_low_clip_rate_increases_lr(self):
        """Test that low clip rate increases learning rate."""
        gamma = 1.0
        new_gamma = clip_rate_based_lr_adjustment(
            current_lr_multiplier=gamma,
            clip_rate=0.05,  # Below low threshold
            target_clip_rate=0.20,
            clip_rate_low=0.10,
            clip_rate_high=0.30,
            increase_factor=1.01,
        )
        assert new_gamma > gamma
        assert new_gamma == pytest.approx(1.01)

    def test_high_clip_rate_decreases_lr(self):
        """Test that high clip rate decreases learning rate."""
        gamma = 1.0
        new_gamma = clip_rate_based_lr_adjustment(
            current_lr_multiplier=gamma,
            clip_rate=0.40,  # Above high threshold
            target_clip_rate=0.20,
            clip_rate_low=0.10,
            clip_rate_high=0.30,
            decrease_factor=0.995,
        )
        assert new_gamma < gamma
        assert new_gamma == pytest.approx(0.995)

    def test_clamping_to_min(self):
        """Test that LR multiplier is clamped to minimum."""
        gamma = 0.15  # Close to min
        new_gamma = clip_rate_based_lr_adjustment(
            current_lr_multiplier=gamma,
            clip_rate=0.50,  # High, will decrease
            target_clip_rate=0.20,
            clip_rate_low=0.10,
            clip_rate_high=0.30,
            decrease_factor=0.5,  # Big decrease
            lr_multiplier_min=0.1,
        )
        assert new_gamma >= 0.1
        assert new_gamma == pytest.approx(0.1)

    def test_clamping_to_max(self):
        """Test that LR multiplier is clamped to maximum."""
        gamma = 1.9  # Close to max
        new_gamma = clip_rate_based_lr_adjustment(
            current_lr_multiplier=gamma,
            clip_rate=0.01,  # Very low, will increase
            target_clip_rate=0.20,
            clip_rate_low=0.10,
            clip_rate_high=0.30,
            increase_factor=2.0,  # Big increase
            lr_multiplier_max=2.0,
        )
        assert new_gamma <= 2.0
        assert new_gamma == pytest.approx(2.0)

    def test_multiple_adjustments_accumulate(self):
        """Test that multiple adjustments accumulate properly."""
        gamma = 1.0

        # Increase 5 times
        for _ in range(5):
            gamma = clip_rate_based_lr_adjustment(
                current_lr_multiplier=gamma,
                clip_rate=0.05,  # Low
                target_clip_rate=0.20,
                clip_rate_low=0.10,
                clip_rate_high=0.30,
                increase_factor=1.01,
            )

        # Should be approximately 1.01^5 ≈ 1.051
        assert gamma == pytest.approx(1.051, abs=0.001)

    def test_extreme_clip_rates(self):
        """Test with extreme clip rates."""
        gamma = 1.0

        # Clip rate = 0 (no clipping)
        new_gamma = clip_rate_based_lr_adjustment(
            current_lr_multiplier=gamma,
            clip_rate=0.0,
            target_clip_rate=0.20,
            clip_rate_low=0.10,
            clip_rate_high=0.30,
        )
        assert new_gamma > gamma

        # Clip rate = 1.0 (all clipping)
        new_gamma = clip_rate_based_lr_adjustment(
            current_lr_multiplier=gamma,
            clip_rate=1.0,
            target_clip_rate=0.20,
            clip_rate_low=0.10,
            clip_rate_high=0.30,
        )
        assert new_gamma < gamma


class TestClipRateThresholdComputation:
    """Tests for computing clip rate thresholds."""

    def test_default_tolerance(self):
        """Test default tolerance computation."""
        low, high = compute_clip_rate_thresholds(0.20, tolerance=0.10)
        assert low == pytest.approx(0.10)
        assert high == pytest.approx(0.30)

    def test_custom_tolerance(self):
        """Test custom tolerance."""
        low, high = compute_clip_rate_thresholds(0.30, tolerance=0.15)
        assert low == pytest.approx(0.15)
        assert high == pytest.approx(0.45)

    def test_clamping_to_zero(self):
        """Test that lower bound is clamped to 0."""
        low, high = compute_clip_rate_thresholds(0.05, tolerance=0.10)
        assert low == 0.0  # max(0, 0.05 - 0.10)
        assert high == pytest.approx(0.15)

    def test_clamping_to_one(self):
        """Test that upper bound is clamped to 1."""
        low, high = compute_clip_rate_thresholds(0.95, tolerance=0.10)
        assert low == pytest.approx(0.85)
        assert high == 1.0  # min(1, 0.95 + 0.10)

    def test_symmetric_around_target(self):
        """Test that thresholds are symmetric around target."""
        target = 0.40
        tolerance = 0.20
        low, high = compute_clip_rate_thresholds(target, tolerance)

        assert (low + high) / 2 == pytest.approx(target)

    def test_zero_tolerance(self):
        """Test with zero tolerance (no range)."""
        low, high = compute_clip_rate_thresholds(0.25, tolerance=0.0)
        assert low == high == 0.25


class TestLRSchedulerEdgeCases:
    """Tests for edge cases in LR scheduling."""

    def test_exactly_at_low_threshold(self):
        """Test behavior when clip rate exactly at low threshold."""
        gamma = 1.0
        new_gamma = clip_rate_based_lr_adjustment(
            current_lr_multiplier=gamma,
            clip_rate=0.10,  # Exactly at low threshold
            target_clip_rate=0.20,
            clip_rate_low=0.10,
            clip_rate_high=0.30,
        )
        # Should be stable (not below, not above)
        assert new_gamma == pytest.approx(1.0)

    def test_exactly_at_high_threshold(self):
        """Test behavior when clip rate exactly at high threshold."""
        gamma = 1.0
        new_gamma = clip_rate_based_lr_adjustment(
            current_lr_multiplier=gamma,
            clip_rate=0.30,  # Exactly at high threshold
            target_clip_rate=0.20,
            clip_rate_low=0.10,
            clip_rate_high=0.30,
        )
        # Should be stable (not below, not above)
        assert new_gamma == pytest.approx(1.0)

    def test_very_small_adjustments(self):
        """Test with very small adjustment factors."""
        gamma = 1.0
        new_gamma = clip_rate_based_lr_adjustment(
            current_lr_multiplier=gamma,
            clip_rate=0.05,
            target_clip_rate=0.20,
            clip_rate_low=0.10,
            clip_rate_high=0.30,
            increase_factor=1.0001,  # Very small increase
        )
        assert new_gamma == pytest.approx(1.0001, abs=1e-6)

    def test_large_adjustments(self):
        """Test with large adjustment factors."""
        gamma = 1.0
        new_gamma = clip_rate_based_lr_adjustment(
            current_lr_multiplier=gamma,
            clip_rate=0.05,
            target_clip_rate=0.20,
            clip_rate_low=0.10,
            clip_rate_high=0.30,
            increase_factor=1.5,  # Large increase
            lr_multiplier_max=2.0,
        )
        assert new_gamma == pytest.approx(1.5)


class TestLRSchedulerIntegration:
    """Integration tests for LR scheduler with realistic scenarios."""

    def test_converge_to_target_from_high(self):
        """Test that scheduler converges toward target from high clip rate."""
        gamma = 1.0
        history = [gamma]

        # Simulate high clip rate scenario
        for _ in range(20):
            gamma = clip_rate_based_lr_adjustment(
                current_lr_multiplier=gamma,
                clip_rate=0.60,  # Consistently high
                target_clip_rate=0.20,
                clip_rate_low=0.10,
                clip_rate_high=0.30,
                decrease_factor=0.99,
            )
            history.append(gamma)

        # LR should decrease over time
        assert history[-1] < history[0]
        # Should be monotonically decreasing
        for i in range(len(history) - 1):
            assert history[i] >= history[i + 1]

    def test_converge_to_target_from_low(self):
        """Test that scheduler converges toward target from low clip rate."""
        gamma = 1.0
        history = [gamma]

        # Simulate low clip rate scenario
        for _ in range(20):
            gamma = clip_rate_based_lr_adjustment(
                current_lr_multiplier=gamma,
                clip_rate=0.02,  # Consistently low
                target_clip_rate=0.20,
                clip_rate_low=0.10,
                clip_rate_high=0.30,
                increase_factor=1.01,
            )
            history.append(gamma)

        # LR should increase over time
        assert history[-1] > history[0]
        # Should be monotonically increasing
        for i in range(len(history) - 1):
            assert history[i] <= history[i + 1]

    def test_stable_at_target(self):
        """Test that scheduler is stable when at target."""
        gamma = 1.0
        history = [gamma]

        # Simulate stable clip rate at target
        for _ in range(10):
            gamma = clip_rate_based_lr_adjustment(
                current_lr_multiplier=gamma,
                clip_rate=0.20,  # At target
                target_clip_rate=0.20,
                clip_rate_low=0.10,
                clip_rate_high=0.30,
            )
            history.append(gamma)

        # Should remain constant
        for val in history:
            assert val == pytest.approx(1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
