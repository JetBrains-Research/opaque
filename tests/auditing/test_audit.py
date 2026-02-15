"""Tests for AuditResult class."""

import numpy as np
import pytest

from opaque.auditing import AuditResult


class TestConstruction:
    """Tests for AuditResult construction."""

    def test_basic_construction(self):
        """Test constructing AuditResult from arrays."""
        result = AuditResult(np.arange(50, 100), np.arange(0, 50))

        assert result.n_in == 50
        assert result.n_out == 50

    def test_list_input(self):
        """Test construction from plain lists."""
        result = AuditResult([1, 2, 3], [4, 5, 6])
        assert result.n_in == 3

    def test_empty_in_scores(self):
        """Test that empty in_scores raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            AuditResult([], [1, 2])

    def test_empty_out_scores(self):
        """Test that empty out_scores raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            AuditResult([1, 2], [])


class TestEpsilonClopperPearson:
    """Tests for epsilon_clopper_pearson method."""

    def test_perfect_separation(self):
        """Test with perfectly separated scores."""
        result = AuditResult(list(range(100, 150)), list(range(0, 50)))

        eps = result.epsilon_clopper_pearson(
            significance=0.05, delta=0, threshold=75
        )
        assert eps > 2.0

    def test_no_separation(self):
        """Test with identical distributions."""
        scores = list(range(100))
        result = AuditResult(scores, scores)

        eps = result.epsilon_clopper_pearson(significance=0.05, delta=0)
        assert eps < 1.0

    def test_invalid_significance(self):
        """Test that invalid significance raises ValueError."""
        result = AuditResult([1, 2], [3, 4])
        with pytest.raises(ValueError, match="significance must be in"):
            result.epsilon_clopper_pearson(significance=0.0)
        with pytest.raises(ValueError, match="significance must be in"):
            result.epsilon_clopper_pearson(significance=0.6)

    def test_invalid_delta(self):
        """Test that invalid delta raises ValueError."""
        result = AuditResult([1, 2], [3, 4])
        with pytest.raises(ValueError, match="delta must be in"):
            result.epsilon_clopper_pearson(significance=0.05, delta=-0.1)
        with pytest.raises(ValueError, match="delta must be in"):
            result.epsilon_clopper_pearson(significance=0.05, delta=1.5)

    def test_explicit_threshold(self):
        """Test with explicit threshold."""
        result = AuditResult(np.arange(50, 100), np.arange(0, 50))

        eps = result.epsilon_clopper_pearson(
            significance=0.05, delta=0, threshold=50
        )
        assert eps > 0


class TestEpsilonOneRun:
    """Tests for epsilon_one_run method."""

    def test_perfect_separation(self):
        """Test with perfectly separated scores."""
        result = AuditResult(list(range(100, 150)), list(range(0, 50)))

        eps = result.epsilon_one_run(
            significance=0.05, delta=0, threshold=75
        )
        assert eps > 0

    def test_no_separation(self):
        """Test with identical distributions."""
        scores = list(range(100))
        result = AuditResult(scores, scores)

        eps = result.epsilon_one_run(significance=0.05, delta=0)
        assert eps < 1.0

    def test_invalid_significance(self):
        """Test that invalid significance raises ValueError."""
        result = AuditResult([1, 2], [3, 4])
        with pytest.raises(ValueError, match="significance must be in"):
            result.epsilon_one_run(significance=0.0)
        with pytest.raises(ValueError, match="significance must be in"):
            result.epsilon_one_run(significance=0.6)

    def test_invalid_delta(self):
        """Test that invalid delta raises ValueError."""
        result = AuditResult([1, 2], [3, 4])
        with pytest.raises(ValueError, match="delta must be in"):
            result.epsilon_one_run(significance=0.05, delta=-0.1)
        with pytest.raises(ValueError, match="delta must be in"):
            result.epsilon_one_run(significance=0.05, delta=1.5)


class TestEpsilonRawCounts:
    """Tests for epsilon_raw_counts method."""

    def test_perfect_attack(self):
        """Test with perfect attack."""
        result = AuditResult(np.arange(50, 100), np.arange(0, 50))

        eps = result.epsilon_raw_counts(min_count=10, delta=0)
        assert eps > 1.5

    def test_no_attack(self):
        """Test with no attack."""
        scores = np.arange(100)
        result = AuditResult(scores, scores)

        eps = result.epsilon_raw_counts(min_count=10, delta=0)
        assert eps == 0.0

    def test_invalid_min_count(self):
        """Test that invalid min_count raises ValueError."""
        result = AuditResult([1, 2], [3, 4])
        with pytest.raises(ValueError, match="min_count must be positive"):
            result.epsilon_raw_counts(min_count=0)
        with pytest.raises(ValueError, match="min_count must be positive"):
            result.epsilon_raw_counts(min_count=-5)

    def test_invalid_delta(self):
        """Test that invalid delta raises ValueError."""
        result = AuditResult([1, 2], [3, 4])
        with pytest.raises(ValueError, match="delta must be in"):
            result.epsilon_raw_counts(delta=-0.1)
        with pytest.raises(ValueError, match="delta must be in"):
            result.epsilon_raw_counts(delta=1.5)


class TestAuroc:
    """Tests for auroc method."""

    def test_perfect_attack(self):
        """Test with perfect attack."""
        result = AuditResult(np.arange(50, 100), np.arange(0, 50))
        assert result.auroc() > 0.99

    def test_random_attack(self):
        """Test with random attack."""
        scores = np.arange(100)
        result = AuditResult(scores, scores)
        assert 0.45 < result.auroc() < 0.55

    def test_negative_scores(self):
        """Test that negative scores work correctly."""
        result = AuditResult(np.arange(-50, 0), np.arange(-100, -50))
        assert result.auroc() > 0.99


class TestTprAtFpr:
    """Tests for tpr_at_fpr method."""

    def test_perfect_classifier(self):
        """Test with perfectly separated scores."""
        result = AuditResult(np.arange(50, 100), np.arange(0, 50))
        assert result.tpr_at_fpr(fpr=0.1) > 0.9

    def test_random_classifier(self):
        """Test with identical distributions."""
        scores = np.arange(100)
        result = AuditResult(scores, scores)
        tpr = result.tpr_at_fpr(fpr=0.1)
        assert 0.05 < tpr < 0.2

    def test_multiple_fprs(self):
        """Test with multiple FPR values."""
        result = AuditResult(np.arange(50, 100), np.arange(0, 50))

        fprs = np.array([0.01, 0.05, 0.1])
        tprs = result.tpr_at_fpr(fpr=fprs)

        assert len(tprs) == 3
        assert np.all(tprs[:-1] <= tprs[1:])

    def test_invalid_fpr(self):
        """Test that invalid FPR raises ValueError."""
        result = AuditResult([1, 2], [3, 4])
        with pytest.raises(ValueError, match="fpr must be in"):
            result.tpr_at_fpr(fpr=-0.1)
        with pytest.raises(ValueError, match="fpr must be in"):
            result.tpr_at_fpr(fpr=1.5)


class TestMaxAccuracy:
    """Tests for max_accuracy method."""

    def test_perfect_classifier(self):
        """Test with perfect separation."""
        result = AuditResult(np.arange(50, 100), np.arange(0, 50))
        assert result.max_accuracy() > 0.99

    def test_random_classifier(self):
        """Test with identical distributions."""
        scores = np.arange(100)
        result = AuditResult(scores, scores)
        assert 0.45 < result.max_accuracy() < 0.55

    def test_custom_prevalence(self):
        """Test with custom prevalence."""
        np.random.seed(42)
        in_scores = np.random.normal(loc=1.0, scale=1.0, size=100)
        out_scores = np.random.normal(loc=0.0, scale=1.0, size=100)
        result = AuditResult(in_scores, out_scores)

        acc_balanced = result.max_accuracy(prevalence=0.5)
        acc_imbalanced = result.max_accuracy(prevalence=0.1)

        assert acc_balanced != acc_imbalanced


class TestEdgeCases:
    """Edge case tests."""

    def test_single_score_each(self):
        """Test with single score in each group."""
        result = AuditResult([10], [0])
        eps = result.epsilon_clopper_pearson(
            significance=0.05, delta=0, threshold=5
        )
        assert eps >= 0

    def test_large_separation(self):
        """Test with very large score separation."""
        result = AuditResult(np.arange(1000, 2000), np.arange(0, 1000))

        eps = result.epsilon_clopper_pearson(
            significance=0.05, delta=0, threshold=1000
        )
        assert eps > 5.0


class TestBootstrap:
    """Tests for bootstrap method."""

    def test_basic_bootstrap(self):
        """Test basic bootstrap functionality."""
        from opaque.auditing import BootstrapParams

        rng = np.random.default_rng(42)
        result = AuditResult(
            rng.normal(2.0, 1.0, 100), rng.normal(0.0, 1.0, 100)
        )

        params = BootstrapParams(num_samples=50, seed=42)
        ci = result.bootstrap(AuditResult.auroc, params)

        assert isinstance(ci, np.ndarray)
        assert len(ci) == 2
        assert ci[0] < ci[1]

    def test_bootstrap_reproducibility(self):
        """Test that bootstrap is reproducible with seed."""
        from opaque.auditing import BootstrapParams

        result = AuditResult(np.arange(50, 100), np.arange(0, 50))
        params = BootstrapParams(num_samples=20, seed=42)

        ci1 = result.bootstrap(AuditResult.auroc, params)
        ci2 = result.bootstrap(AuditResult.auroc, params)

        np.testing.assert_array_equal(ci1, ci2)

    def test_bootstrap_custom_quantiles(self):
        """Test bootstrap with custom quantiles."""
        from opaque.auditing import BootstrapParams

        rng = np.random.default_rng(42)
        result = AuditResult(
            rng.normal(2.0, 1.0, 100), rng.normal(0.0, 1.0, 100)
        )

        params = BootstrapParams(
            num_samples=50, quantiles=(0.1, 0.5, 0.9), seed=42
        )
        ci = result.bootstrap(AuditResult.auroc, params)

        assert len(ci) == 3
        assert ci[0] <= ci[1] <= ci[2]

    def test_bootstrap_with_lambda(self):
        """Test bootstrap with lambda for parameterized metrics."""
        from opaque.auditing import BootstrapParams

        result = AuditResult(np.arange(50, 100), np.arange(0, 50))
        params = BootstrapParams(num_samples=20, seed=42)

        ci = result.bootstrap(
            lambda r: r.epsilon_clopper_pearson(significance=0.05), params
        )
        assert len(ci) == 2
