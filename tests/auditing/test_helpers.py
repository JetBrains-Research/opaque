"""Tests for auditing helper functions."""

import numpy as np
import pytest

from opaque.auditing.helpers import (
    _clopper_pearson_upper,
    _epsilon_raw_counts_helper,
    _get_tn_fn_counts,
    _log_sub,
    _one_run_p_value,
    _pareto_frontier,
    _random_partition,
    _tpr_at_given_fpr,
)


class TestLogSub:
    def test_basic_computation(self):
        x = np.log(10)
        y = np.log(3)
        result = _log_sub(x, y)
        expected = np.log(10 - 3)
        assert np.isclose(result, expected)

    def test_near_equal_values(self):
        x = np.log(1.0001)
        y = np.log(1.0)
        result = _log_sub(x, y)
        expected = np.log(0.0001)
        assert np.isclose(result, expected, rtol=1e-3)

    def test_equal_values(self):
        x = np.log(5)
        y = np.log(5)
        result = _log_sub(x, y)
        assert result == -np.inf

    def test_invalid_order(self):
        with pytest.raises(ValueError, match="y must be <= x"):
            _log_sub(np.log(3), np.log(10))


class TestClopperPearsonUpper:
    def test_zero_successes(self):
        result = _clopper_pearson_upper(0, 100, 0.05)
        assert 0 < result < 0.05

    def test_all_successes(self):
        result = _clopper_pearson_upper(100, 100, 0.05)
        assert result == 1.0

    def test_half_successes(self):
        result = _clopper_pearson_upper(50, 100, 0.05)
        assert 0.5 < result < 0.6

    def test_vectorized(self):
        k = np.array([0, 25, 50, 75, 100])
        result = _clopper_pearson_upper(k, 100, 0.05)
        assert len(result) == 5
        assert np.all(result[:-1] <= result[1:])


class TestParetoFrontier:
    def test_simple_frontier(self):
        points = np.array([[0, 0], [1, 2], [2, 1], [3, 3]])
        indices = _pareto_frontier(points)
        expected = np.array([0, 1, 3])
        np.testing.assert_array_equal(indices, expected)

    def test_all_on_line(self):
        points = np.array([[0, 0], [1, 1], [2, 2]])
        indices = _pareto_frontier(points)
        expected = np.array([0, 2])
        np.testing.assert_array_equal(indices, expected)

    def test_only_two_points(self):
        points = np.array([[0, 0], [1, 1]])
        indices = _pareto_frontier(points)
        expected = np.array([0, 1])
        np.testing.assert_array_equal(indices, expected)

    def test_invalid_shape(self):
        with pytest.raises(ValueError, match="Expected at least two 2D points"):
            _pareto_frontier(np.array([[0, 0, 0]]))

    def test_unsorted_raises(self):
        with pytest.raises(ValueError, match="Expected points to be sorted"):
            _pareto_frontier(np.array([[1, 1], [0, 0]]))


class TestGetTnFnCounts:
    def test_perfect_separation(self):
        thresholds, tn, fn = _get_tn_fn_counts([5, 6, 7, 8, 9], [0, 1, 2, 3, 4])
        idx = np.where(thresholds == 5)[0]
        if len(idx) > 0:
            assert tn[idx[0]] == 5
            assert fn[idx[0]] == 0

    def test_complete_overlap(self):
        thresholds, tn, fn = _get_tn_fn_counts([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(tn, fn)

    def test_empty_out_scores(self):
        thresholds, tn, fn = _get_tn_fn_counts([1, 2, 3], [])
        assert np.all(tn == 0)

    def test_both_empty_raises(self):
        with pytest.raises(ValueError, match="must be non-empty"):
            _get_tn_fn_counts([], [])


class TestTprAtGivenFpr:
    def test_perfect_classifier(self):
        tp_counts = np.array([0, 100])
        fp_counts = np.array([0, 1])
        tpr = _tpr_at_given_fpr(0.0, tp_counts, fp_counts)
        assert tpr == 0.0

    def test_random_classifier(self):
        tp_counts = np.array([0, 50, 100])
        fp_counts = np.array([0, 50, 100])
        tpr = _tpr_at_given_fpr(0.5, tp_counts, fp_counts)
        assert np.isclose(tpr, 0.5)

    def test_vectorized_fpr(self):
        tp_counts = np.array([0, 50, 100])
        fp_counts = np.array([0, 50, 100])
        fprs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        tprs = _tpr_at_given_fpr(fprs, tp_counts, fp_counts)
        assert len(tprs) == len(fprs)
        assert np.all(tprs[:-1] <= tprs[1:])

    def test_invalid_fpr(self):
        tp_counts = np.array([0, 100])
        fp_counts = np.array([0, 100])
        with pytest.raises(ValueError, match="fpr must be in"):
            _tpr_at_given_fpr(-0.1, tp_counts, fp_counts)
        with pytest.raises(ValueError, match="fpr must be in"):
            _tpr_at_given_fpr(1.5, tp_counts, fp_counts)


class TestEpsilonRawCountsHelper:
    def test_no_attack(self):
        tp_counts = np.array([0, 50, 100])
        fp_counts = np.array([0, 50, 100])
        eps = _epsilon_raw_counts_helper(tp_counts, fp_counts, min_count=10, delta=0)
        assert eps == 0.0

    def test_perfect_attack(self):
        tp_counts = np.array([0, 50, 90, 100])
        fp_counts = np.array([0, 5, 10, 20])
        eps = _epsilon_raw_counts_helper(tp_counts, fp_counts, min_count=5, delta=0)
        assert eps > 0.5

    def test_with_delta(self):
        tp_counts = np.array([0, 50, 100])
        fp_counts = np.array([0, 10, 20])
        eps_delta0 = _epsilon_raw_counts_helper(
            tp_counts, fp_counts, min_count=5, delta=0
        )
        eps_delta = _epsilon_raw_counts_helper(
            tp_counts, fp_counts, min_count=5, delta=0.01
        )
        assert eps_delta >= eps_delta0

    def test_min_count_threshold(self):
        tp_counts = np.array([0, 90, 100])
        fp_counts = np.array([0, 5, 100])
        eps = _epsilon_raw_counts_helper(tp_counts, fp_counts, min_count=200, delta=0)
        assert eps == 0.0


class TestRandomPartition:
    def test_partition_sizes(self):
        scores = np.arange(100)
        rng = np.random.default_rng(42)
        part1, part2 = _random_partition(scores, rng, 0.3)
        assert len(part1) == 30
        assert len(part2) == 70

    def test_no_overlap(self):
        scores = np.arange(100)
        rng = np.random.default_rng(42)
        part1, part2 = _random_partition(scores, rng, 0.5)
        assert len(np.intersect1d(part1, part2)) == 0

    def test_invalid_p(self):
        scores = np.arange(10)
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="p must be in"):
            _random_partition(scores, rng, 0.0)
        with pytest.raises(ValueError, match="p must be in"):
            _random_partition(scores, rng, 1.0)


class TestOneRunPValue:
    def test_zero_epsilon(self):
        p_value = _one_run_p_value(m=100, n_guess=50, n_correct=25, eps=0, delta=0)
        assert 0.4 < p_value < 0.6

    def test_small_epsilon(self):
        p_value = _one_run_p_value(m=100, n_guess=50, n_correct=30, eps=0.5, delta=0)
        assert 0.1 < p_value < 1.0

    def test_with_delta(self):
        p_delta0 = _one_run_p_value(m=100, n_guess=50, n_correct=30, eps=1.0, delta=0)
        p_delta = _one_run_p_value(m=100, n_guess=50, n_correct=30, eps=1.0, delta=0.01)
        assert p_delta >= p_delta0
