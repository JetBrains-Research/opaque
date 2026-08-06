"""Tests for auditing ROC helpers (one_run/roc.py)."""

import numpy as np
import pytest

from opaque.api.auditing.one_run._roc import (
    get_tn_fn_counts,
    pareto_frontier,
    tpr_at_given_fpr,
)


class TestParetoFrontier:
    def test_simple_frontier(self):
        points = np.array([[0, 0], [1, 2], [2, 1], [3, 3]])
        indices = pareto_frontier(points)
        expected = np.array([0, 1, 3])
        np.testing.assert_array_equal(indices, expected)

    def test_all_on_line(self):
        points = np.array([[0, 0], [1, 1], [2, 2]])
        indices = pareto_frontier(points)
        expected = np.array([0, 2])
        np.testing.assert_array_equal(indices, expected)

    def test_only_two_points(self):
        points = np.array([[0, 0], [1, 1]])
        indices = pareto_frontier(points)
        expected = np.array([0, 1])
        np.testing.assert_array_equal(indices, expected)

    def test_invalid_shape(self):
        with pytest.raises(ValueError, match="Expected at least two 2D points"):
            pareto_frontier(np.array([[0, 0, 0]]))

    def test_unsorted_raises(self):
        with pytest.raises(ValueError, match="Expected points to be sorted"):
            pareto_frontier(np.array([[1, 1], [0, 0]]))


class TestGetTnFnCounts:
    def test_perfect_separation(self):
        thresholds, tn, fn = get_tn_fn_counts([5, 6, 7, 8, 9], [0, 1, 2, 3, 4])
        idx = np.where(thresholds == 5)[0]
        if len(idx) > 0:
            assert tn[idx[0]] == 5
            assert fn[idx[0]] == 0

    def test_complete_overlap(self):
        _thresholds, tn, fn = get_tn_fn_counts([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(tn, fn)

    def test_empty_out_scores(self):
        _thresholds, tn, _fn = get_tn_fn_counts([1, 2, 3], [])
        assert np.all(tn == 0)

    def test_both_empty_raises(self):
        with pytest.raises(ValueError, match="must be non-empty"):
            get_tn_fn_counts([], [])


class TestTprAtGivenFpr:
    def test_perfect_classifier(self):
        tp_counts = np.array([0, 100])
        fp_counts = np.array([0, 1])
        tpr = tpr_at_given_fpr(0.0, tp_counts, fp_counts)
        assert tpr == 0.0

    def test_random_classifier(self):
        tp_counts = np.array([0, 50, 100])
        fp_counts = np.array([0, 50, 100])
        tpr = tpr_at_given_fpr(0.5, tp_counts, fp_counts)
        assert np.isclose(tpr, 0.5)

    def test_vectorized_fpr(self):
        tp_counts = np.array([0, 50, 100])
        fp_counts = np.array([0, 50, 100])
        fprs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        tprs = tpr_at_given_fpr(fprs, tp_counts, fp_counts)
        assert len(tprs) == len(fprs)
        assert np.all(tprs[:-1] <= tprs[1:])

    def test_invalid_fpr(self):
        tp_counts = np.array([0, 100])
        fp_counts = np.array([0, 100])
        with pytest.raises(ValueError, match="fpr must be in"):
            tpr_at_given_fpr(-0.1, tp_counts, fp_counts)
        with pytest.raises(ValueError, match="fpr must be in"):
            tpr_at_given_fpr(1.5, tp_counts, fp_counts)


def test_raw_auc_is_unbiased_under_null():
    """#378: raw-ROC AUC recovers ~0.5 under the null; the hull basis is biased high."""
    from opaque.api.auditing.one_run._estimate import _auc_from_counts

    rng = np.random.default_rng(0)
    n, trials = 64, 300
    raw, hull = [], []
    for _ in range(trials):
        a = rng.standard_normal(n)
        b = rng.standard_normal(n)
        _, tn, fn = get_tn_fn_counts(a, b)
        raw.append(_auc_from_counts(tn, fn))
        _, htn, hfn = get_tn_fn_counts(a, b, hull=True)
        hull.append(_auc_from_counts(htn, hfn))
    raw_mean, hull_mean = float(np.mean(raw)), float(np.mean(hull))
    assert abs(raw_mean - 0.5) < 0.02, raw_mean
    assert hull_mean > raw_mean + 0.01, (raw_mean, hull_mean)


def test_infinite_scores_counted_in_denominators():
    """#378: +inf scores are included in the n_in / n_out ROC denominators."""
    _, tn, fn = get_tn_fn_counts([1.0, 2.0, np.inf], [0.5, 1.5, 2.5])
    assert fn[-1] == 3  # n_in (includes the +inf in-score), not the finite-only 2
    assert tn[-1] == 3  # n_out
