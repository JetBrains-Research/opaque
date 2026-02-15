"""Tests for epsilon estimation functions."""

import numpy as np
import pytest

from opaque.auditing import (
    epsilon_clopper_pearson,
    epsilon_one_run,
    epsilon_raw_counts,
)


class TestEpsilonClopperPearson:
    """Tests for epsilon_clopper_pearson function."""

    def test_perfect_separation(self):
        """Test with perfectly separated scores."""
        in_scores = list(range(100, 150))
        out_scores = list(range(0, 50))

        eps = epsilon_clopper_pearson(
            in_scores, out_scores, significance=0.05, delta=0, threshold=75
        )
        assert eps > 2.0

    def test_no_separation(self):
        """Test with identical distributions."""
        scores = list(range(100))

        eps = epsilon_clopper_pearson(scores, scores, significance=0.05, delta=0)
        assert eps < 1.0

    def test_invalid_significance(self):
        """Test that invalid significance raises ValueError."""
        with pytest.raises(ValueError, match="significance must be in"):
            epsilon_clopper_pearson([1, 2], [3, 4], significance=0.0)
        with pytest.raises(ValueError, match="significance must be in"):
            epsilon_clopper_pearson([1, 2], [3, 4], significance=0.6)

    def test_invalid_delta(self):
        """Test that invalid delta raises ValueError."""
        with pytest.raises(ValueError, match="delta must be in"):
            epsilon_clopper_pearson([1, 2], [3, 4], significance=0.05, delta=-0.1)
        with pytest.raises(ValueError, match="delta must be in"):
            epsilon_clopper_pearson([1, 2], [3, 4], significance=0.05, delta=1.5)

    def test_explicit_threshold(self):
        """Test with explicit threshold."""
        in_scores = np.arange(50, 100)
        out_scores = np.arange(0, 50)

        eps = epsilon_clopper_pearson(
            in_scores, out_scores, significance=0.05, delta=0, threshold=50
        )
        assert eps > 0


class TestEpsilonOneRun:
    """Tests for epsilon_one_run function."""

    def test_perfect_separation(self):
        """Test with perfectly separated scores."""
        in_scores = list(range(100, 150))
        out_scores = list(range(0, 50))

        eps = epsilon_one_run(
            in_scores, out_scores, significance=0.05, delta=0, threshold=75
        )
        assert eps > 0

    def test_no_separation(self):
        """Test with identical distributions."""
        scores = list(range(100))

        eps = epsilon_one_run(scores, scores, significance=0.05, delta=0)
        assert eps < 1.0

    def test_invalid_significance(self):
        """Test that invalid significance raises ValueError."""
        with pytest.raises(ValueError, match="significance must be in"):
            epsilon_one_run([1, 2], [3, 4], significance=0.0)
        with pytest.raises(ValueError, match="significance must be in"):
            epsilon_one_run([1, 2], [3, 4], significance=0.6)

    def test_invalid_delta(self):
        """Test that invalid delta raises ValueError."""
        with pytest.raises(ValueError, match="delta must be in"):
            epsilon_one_run([1, 2], [3, 4], significance=0.05, delta=-0.1)
        with pytest.raises(ValueError, match="delta must be in"):
            epsilon_one_run([1, 2], [3, 4], significance=0.05, delta=1.5)


class TestEpsilonRawCounts:
    """Tests for epsilon_raw_counts function."""

    def test_perfect_attack(self):
        """Test with perfect attack."""
        in_scores = np.arange(50, 100)
        out_scores = np.arange(0, 50)

        eps = epsilon_raw_counts(in_scores, out_scores, min_count=10, delta=0)
        assert eps > 1.5

    def test_no_attack(self):
        """Test with no attack."""
        scores = np.arange(100)

        eps = epsilon_raw_counts(scores, scores, min_count=10, delta=0)
        assert eps == 0.0

    def test_invalid_min_count(self):
        """Test that invalid min_count raises ValueError."""
        with pytest.raises(ValueError, match="min_count must be positive"):
            epsilon_raw_counts([1, 2], [3, 4], min_count=0)
        with pytest.raises(ValueError, match="min_count must be positive"):
            epsilon_raw_counts([1, 2], [3, 4], min_count=-5)

    def test_invalid_delta(self):
        """Test that invalid delta raises ValueError."""
        with pytest.raises(ValueError, match="delta must be in"):
            epsilon_raw_counts([1, 2], [3, 4], delta=-0.1)
        with pytest.raises(ValueError, match="delta must be in"):
            epsilon_raw_counts([1, 2], [3, 4], delta=1.5)


class TestEdgeCases:
    """Edge case tests for epsilon functions."""

    def test_single_score_each(self):
        """Test with single score in each group."""
        eps = epsilon_clopper_pearson(
            [10], [0], significance=0.05, delta=0, threshold=5
        )
        assert eps >= 0

    def test_large_separation(self):
        """Test with very large score separation."""
        in_scores = np.arange(1000, 2000)
        out_scores = np.arange(0, 1000)

        eps = epsilon_clopper_pearson(
            in_scores, out_scores, significance=0.05, delta=0, threshold=1000
        )
        assert eps > 5.0
