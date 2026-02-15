"""Tests for auditing functions."""

import numpy as np
import pytest

from opaque.auditing import (
    AuditResult,
    BootstrapParams,
    attack_auroc,
    audit,
    bootstrap,
    epsilon_clopper_pearson,
    epsilon_one_run,
    epsilon_raw_counts,
    max_accuracy,
    tpr_at_fpr,
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


class TestAudit:
    """Tests for audit convenience function."""

    def test_returns_namedtuple(self):
        """Test that audit returns an AuditResult."""
        in_scores = np.arange(50, 100)
        out_scores = np.arange(0, 50)

        result = audit(in_scores, out_scores, significance=0.05, delta=0)

        assert isinstance(result, AuditResult)
        assert hasattr(result, "epsilon")
        assert hasattr(result, "auroc")
        assert hasattr(result, "tpr_at_low_fpr")
        assert hasattr(result, "max_accuracy")

    def test_clopper_pearson_method(self):
        """Test with Clopper-Pearson method."""
        in_scores = np.arange(50, 100)
        out_scores = np.arange(0, 50)

        result = audit(in_scores, out_scores, method="clopper_pearson")

        assert result.epsilon > 0
        assert result.auroc > 0.99

    def test_raw_counts_method(self):
        """Test with raw_counts method."""
        in_scores = np.arange(100, 200)
        out_scores = np.arange(0, 100)

        result = audit(in_scores, out_scores, method="raw_counts")

        assert result.epsilon > 0
        assert result.auroc > 0.99

    def test_one_run_method(self):
        """Test with one_run method."""
        in_scores = np.arange(50, 100)
        out_scores = np.arange(0, 50)

        result = audit(in_scores, out_scores, method="one_run")

        assert result.epsilon > 0
        assert result.auroc > 0.99

    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            audit([1, 2], [3, 4], method="invalid")

    def test_unpacking(self):
        """Test that result can be unpacked."""
        in_scores = np.arange(50, 100)
        out_scores = np.arange(0, 50)

        eps, auroc, tpr, acc = audit(in_scores, out_scores)

        assert eps > 0
        assert auroc > 0.99


class TestBootstrap:
    """Tests for bootstrap function."""

    def test_basic_bootstrap(self):
        """Test basic bootstrap functionality."""
        rng = np.random.default_rng(42)
        in_scores = rng.normal(2.0, 1.0, 100)
        out_scores = rng.normal(0.0, 1.0, 100)

        params = BootstrapParams(num_samples=50, seed=42)
        result = bootstrap(attack_auroc, in_scores, out_scores, params)

        assert isinstance(result, np.ndarray)
        assert len(result) == 2
        assert result[0] < result[1]

    def test_bootstrap_reproducibility(self):
        """Test that bootstrap is reproducible with seed."""
        in_scores = np.arange(50, 100)
        out_scores = np.arange(0, 50)

        params = BootstrapParams(num_samples=20, seed=42)
        result1 = bootstrap(attack_auroc, in_scores, out_scores, params)
        result2 = bootstrap(attack_auroc, in_scores, out_scores, params)

        np.testing.assert_array_equal(result1, result2)

    def test_bootstrap_custom_quantiles(self):
        """Test bootstrap with custom quantiles."""
        rng = np.random.default_rng(42)
        in_scores = rng.normal(2.0, 1.0, 100)
        out_scores = rng.normal(0.0, 1.0, 100)

        params = BootstrapParams(num_samples=50, quantiles=(0.1, 0.5, 0.9), seed=42)
        result = bootstrap(attack_auroc, in_scores, out_scores, params)

        assert len(result) == 3
        assert result[0] <= result[1] <= result[2]


class TestEdgeCases:
    """Tests for edge cases."""

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

    def test_negative_scores(self):
        """Test that negative scores work correctly."""
        in_scores = np.arange(-50, 0)
        out_scores = np.arange(-100, -50)

        auroc = attack_auroc(in_scores, out_scores)
        assert auroc > 0.99
