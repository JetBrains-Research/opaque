"""Tests for attack utility metrics."""

import numpy as np
import pytest

from opaque.auditing import attack_auroc, max_accuracy, tpr_at_fpr


class TestAttackAuroc:
    """Tests for attack_auroc function."""

    def test_perfect_attack(self):
        """Test with perfect attack."""
        in_scores = np.arange(50, 100)
        out_scores = np.arange(0, 50)

        auroc = attack_auroc(in_scores, out_scores)
        assert auroc > 0.99

    def test_random_attack(self):
        """Test with random attack."""
        scores = np.arange(100)

        auroc = attack_auroc(scores, scores)
        assert 0.45 < auroc < 0.55

    def test_negative_scores(self):
        """Test that negative scores work correctly."""
        in_scores = np.arange(-50, 0)
        out_scores = np.arange(-100, -50)

        auroc = attack_auroc(in_scores, out_scores)
        assert auroc > 0.99


class TestTprAtFpr:
    """Tests for tpr_at_fpr function."""

    def test_perfect_classifier(self):
        """Test with perfectly separated scores."""
        in_scores = np.arange(50, 100)
        out_scores = np.arange(0, 50)

        tpr = tpr_at_fpr(in_scores, out_scores, fpr=0.1)
        assert tpr > 0.9

    def test_random_classifier(self):
        """Test with identical distributions."""
        scores = np.arange(100)

        tpr = tpr_at_fpr(scores, scores, fpr=0.1)
        assert 0.05 < tpr < 0.2

    def test_multiple_fprs(self):
        """Test with multiple FPR values."""
        in_scores = np.arange(50, 100)
        out_scores = np.arange(0, 50)

        fprs = np.array([0.01, 0.05, 0.1])
        tprs = tpr_at_fpr(in_scores, out_scores, fpr=fprs)

        assert len(tprs) == 3
        assert np.all(tprs[:-1] <= tprs[1:])

    def test_invalid_fpr(self):
        """Test that invalid FPR raises ValueError."""
        with pytest.raises(ValueError, match="fpr must be in"):
            tpr_at_fpr([1, 2], [3, 4], fpr=-0.1)
        with pytest.raises(ValueError, match="fpr must be in"):
            tpr_at_fpr([1, 2], [3, 4], fpr=1.5)


class TestMaxAccuracy:
    """Tests for max_accuracy function."""

    def test_perfect_classifier(self):
        """Test with perfect separation."""
        in_scores = np.arange(50, 100)
        out_scores = np.arange(0, 50)

        acc = max_accuracy(in_scores, out_scores)
        assert acc > 0.99

    def test_random_classifier(self):
        """Test with identical distributions."""
        scores = np.arange(100)

        acc = max_accuracy(scores, scores)
        assert 0.45 < acc < 0.55

    def test_custom_prevalence(self):
        """Test with custom prevalence."""
        # Use overlapping distributions where prevalence affects accuracy
        np.random.seed(42)
        in_scores = np.random.normal(loc=1.0, scale=1.0, size=100)
        out_scores = np.random.normal(loc=0.0, scale=1.0, size=100)

        acc_balanced = max_accuracy(in_scores, out_scores, prevalence=0.5)
        acc_imbalanced = max_accuracy(in_scores, out_scores, prevalence=0.1)

        # With overlapping distributions, prevalence affects weighted accuracy
        assert acc_balanced != acc_imbalanced
