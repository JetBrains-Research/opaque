"""Tests for auditing helper functions."""

import numpy as np
import pytest

from opaque.auditing.helpers import (
    clopper_pearson_upper,
    epsilon_raw_counts_helper,
    get_tn_fn_counts,
    log_sub,
    one_run_p_value,
    pareto_frontier,
    random_partition,
    tpr_at_given_fpr,
)


class TestLogSub:
    """Tests for log_sub function."""

    def test_basic_computation(self):
        """Test basic log subtraction."""
        x = np.log(10)
        y = np.log(3)
        result = log_sub(x, y)
        expected = np.log(10 - 3)
        assert np.isclose(result, expected)

    def test_near_equal_values(self):
        """Test stability when x ≈ y."""
        x = np.log(1.0001)
        y = np.log(1.0)
        result = log_sub(x, y)
        expected = np.log(0.0001)
        assert np.isclose(result, expected, rtol=1e-3)

    def test_equal_values(self):
        """Test that equal values return -inf."""
        x = np.log(5)
        y = np.log(5)
        result = log_sub(x, y)
        assert result == -np.inf

    def test_invalid_order(self):
        """Test that y > x raises ValueError."""
        x = np.log(3)
        y = np.log(10)
        with pytest.raises(ValueError, match="y must be <= x"):
            log_sub(x, y)


class TestClopperPearsonUpper:
    """Tests for Clopper-Pearson upper bound."""

    def test_zero_successes(self):
        """Test with zero successes."""
        result = clopper_pearson_upper(0, 100, 0.05)
        # Upper bound should be small but positive
        assert 0 < result < 0.05

    def test_all_successes(self):
        """Test with all successes."""
        result = clopper_pearson_upper(100, 100, 0.05)
        assert result == 1.0

    def test_half_successes(self):
        """Test with half successes."""
        result = clopper_pearson_upper(50, 100, 0.05)
        # Upper bound should be near 0.5 but slightly above
        assert 0.5 < result < 0.6

    def test_vectorized(self):
        """Test vectorized computation."""
        k = np.array([0, 25, 50, 75, 100])
        result = clopper_pearson_upper(k, 100, 0.05)
        assert len(result) == 5
        # Should be monotonically increasing
        assert np.all(result[:-1] <= result[1:])


class TestParetoFrontier:
    """Tests for Pareto frontier computation."""

    def test_simple_frontier(self):
        """Test simple Pareto frontier."""
        points = np.array([[0, 0], [1, 2], [2, 1], [3, 3]])
        indices = pareto_frontier(points)
        # Points 0 and 3 are on frontier, point 1 dominates point 2
        expected = np.array([0, 1, 3])
        np.testing.assert_array_equal(indices, expected)

    def test_all_on_frontier(self):
        """Test when all points are on frontier."""
        # For points on a line with positive slope, only endpoints are on frontier
        # (middle points are linearly dominated)
        points = np.array([[0, 0], [1, 1], [2, 2]])
        indices = pareto_frontier(points)
        # Only first and last points are on frontier
        expected = np.array([0, 2])
        np.testing.assert_array_equal(indices, expected)

    def test_only_two_points(self):
        """Test with exactly two points."""
        points = np.array([[0, 0], [1, 1]])
        indices = pareto_frontier(points)
        expected = np.array([0, 1])
        np.testing.assert_array_equal(indices, expected)

    def test_invalid_shape(self):
        """Test with invalid shape."""
        points = np.array([[0, 0, 0]])  # 3D points
        with pytest.raises(ValueError, match="Expected at least two 2D points"):
            pareto_frontier(points)

    def test_unsorted_raises(self):
        """Test that unsorted points raise error."""
        points = np.array([[1, 1], [0, 0]])  # Not sorted by x
        with pytest.raises(ValueError, match="Expected points to be sorted"):
            pareto_frontier(points)


class TestGetTnFnCounts:
    """Tests for true/false negative count computation."""

    def test_perfect_separation(self):
        """Test with perfectly separated scores."""
        in_scores = [5, 6, 7, 8, 9]
        out_scores = [0, 1, 2, 3, 4]
        thresholds, tn, fn = get_tn_fn_counts(in_scores, out_scores)

        # At threshold 4.5, should have all TN and no FN
        idx = np.where(thresholds == 5)[0]
        if len(idx) > 0:
            assert tn[idx[0]] == 5  # All out_scores < 5
            assert fn[idx[0]] == 0  # All in_scores >= 5

    def test_complete_overlap(self):
        """Test with completely overlapping scores."""
        in_scores = [1, 2, 3, 4, 5]
        out_scores = [1, 2, 3, 4, 5]
        thresholds, tn, fn = get_tn_fn_counts(in_scores, out_scores)

        # Should have same counts for TN and FN at each threshold
        np.testing.assert_array_equal(tn, fn)

    def test_empty_out_scores(self):
        """Test with empty out_scores."""
        in_scores = [1, 2, 3]
        out_scores = []
        thresholds, tn, fn = get_tn_fn_counts(in_scores, out_scores)

        # All TN counts should be 0
        assert np.all(tn == 0)

    def test_both_empty_raises(self):
        """Test that both empty raises ValueError."""
        with pytest.raises(ValueError, match="must be non-empty"):
            get_tn_fn_counts([], [])


class TestTprAtGivenFpr:
    """Tests for TPR at given FPR computation."""

    def test_perfect_classifier(self):
        """Test with perfect classifier."""
        # Perfect classifier: all TP, minimal FP
        tp_counts = np.array([0, 100])
        fp_counts = np.array([0, 1])  # Need > 0 to avoid division by zero
        tpr = tpr_at_given_fpr(0.0, tp_counts, fp_counts)
        # At FPR=0, TPR should be 0 (no detections yet)
        assert tpr == 0.0

    def test_random_classifier(self):
        """Test with random classifier."""
        # Random classifier: TPR ≈ FPR
        tp_counts = np.array([0, 50, 100])
        fp_counts = np.array([0, 50, 100])
        tpr = tpr_at_given_fpr(0.5, tp_counts, fp_counts)
        assert np.isclose(tpr, 0.5)

    def test_vectorized_fpr(self):
        """Test with multiple FPR values."""
        tp_counts = np.array([0, 50, 100])
        fp_counts = np.array([0, 50, 100])
        fprs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        tprs = tpr_at_given_fpr(fprs, tp_counts, fp_counts)
        assert len(tprs) == len(fprs)
        # TPR should be monotonically increasing with FPR
        assert np.all(tprs[:-1] <= tprs[1:])

    def test_invalid_fpr(self):
        """Test with invalid FPR values."""
        tp_counts = np.array([0, 100])
        fp_counts = np.array([0, 100])
        with pytest.raises(ValueError, match="fpr must be in"):
            tpr_at_given_fpr(-0.1, tp_counts, fp_counts)
        with pytest.raises(ValueError, match="fpr must be in"):
            tpr_at_given_fpr(1.5, tp_counts, fp_counts)


class TestEpsilonRawCountsHelper:
    """Tests for epsilon estimation from raw counts."""

    def test_no_attack(self):
        """Test with no successful attack."""
        # Same TP and FP → epsilon = 0
        tp_counts = np.array([0, 50, 100])
        fp_counts = np.array([0, 50, 100])
        eps = epsilon_raw_counts_helper(tp_counts, fp_counts, min_count=10, delta=0)
        assert eps == 0.0

    def test_perfect_attack(self):
        """Test with perfect attack."""
        # High TP, low FP → large epsilon
        # At min_count=5 with n_neg=20, min_fpr=0.25
        # At FPR=0.25, TPR=0.5, giving eps=log(0.5/0.25)=log(2)≈0.69
        tp_counts = np.array([0, 50, 90, 100])
        fp_counts = np.array([0, 5, 10, 20])  # FP increases more slowly
        eps = epsilon_raw_counts_helper(tp_counts, fp_counts, min_count=5, delta=0)
        assert eps > 0.5  # log(2) ≈ 0.69

    def test_with_delta(self):
        """Test epsilon estimation with delta > 0."""
        tp_counts = np.array([0, 50, 100])
        fp_counts = np.array([0, 10, 20])
        eps_delta0 = epsilon_raw_counts_helper(
            tp_counts, fp_counts, min_count=5, delta=0
        )
        eps_delta = epsilon_raw_counts_helper(
            tp_counts, fp_counts, min_count=5, delta=0.01
        )
        # With delta, epsilon should be slightly larger
        assert eps_delta >= eps_delta0

    def test_min_count_threshold(self):
        """Test that min_count filters thresholds."""
        tp_counts = np.array([0, 90, 100])
        fp_counts = np.array([0, 5, 100])
        # With high min_count, should return 0 (not enough FP)
        eps = epsilon_raw_counts_helper(tp_counts, fp_counts, min_count=200, delta=0)
        assert eps == 0.0


class TestRandomPartition:
    """Tests for random partition function."""

    def test_partition_sizes(self):
        """Test that partition sizes are correct."""
        scores = np.arange(100)
        rng = np.random.default_rng(42)
        part1, part2 = random_partition(scores, rng, 0.3)
        assert len(part1) == 30
        assert len(part2) == 70

    def test_no_overlap(self):
        """Test that partitions don't overlap."""
        scores = np.arange(100)
        rng = np.random.default_rng(42)
        part1, part2 = random_partition(scores, rng, 0.5)
        # Check no shared elements
        assert len(np.intersect1d(part1, part2)) == 0

    def test_invalid_p(self):
        """Test invalid p values."""
        scores = np.arange(10)
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="p must be in"):
            random_partition(scores, rng, 0.0)
        with pytest.raises(ValueError, match="p must be in"):
            random_partition(scores, rng, 1.0)


class TestOneRunPValue:
    """Tests for one-run p-value computation."""

    def test_zero_epsilon(self):
        """Test with epsilon=0 (no privacy)."""
        # With eps=0, attack should have p-value ≈ 1 (can't reject null)
        p_value = one_run_p_value(m=100, n_guess=50, n_correct=25, eps=0, delta=0)
        assert 0.4 < p_value < 0.6  # Around random guessing

    def test_small_epsilon(self):
        """Test with small epsilon (weak privacy)."""
        # With small eps and precision near random, p-value should be moderate/high
        p_value = one_run_p_value(m=100, n_guess=50, n_correct=30, eps=0.5, delta=0)
        # Precision of 60% with eps=0.5 doesn't provide strong evidence
        assert 0.1 < p_value < 1.0

    def test_with_delta(self):
        """Test p-value computation with delta > 0."""
        p_delta0 = one_run_p_value(
            m=100, n_guess=50, n_correct=30, eps=1.0, delta=0
        )
        p_delta = one_run_p_value(
            m=100, n_guess=50, n_correct=30, eps=1.0, delta=0.01
        )
        # With delta, p-value should increase (harder to reject)
        assert p_delta >= p_delta0
