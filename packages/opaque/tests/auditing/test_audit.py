"""Tests for AuditResult, CoinFlip, and OneRunEstimator classes."""

import numpy as np
import pytest

import opaque.auditing as auditing
from opaque.auditing import AuditResult, CoinFlip, OneRunEstimator
from opaque.random import key


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


class TestEpsilonOneRun:
    """Tests for epsilon_one_run method."""

    def test_perfect_separation(self):
        """Test with perfectly separated scores."""
        result = AuditResult(list(range(100, 150)), list(range(0, 50)))

        eps = result.epsilon_one_run(significance=0.05, delta=0, threshold=75)
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


class TestAuc:
    """Tests for auc method."""

    def test_perfect_attack(self):
        """Test with perfect attack."""
        result = AuditResult(np.arange(50, 100), np.arange(0, 50))
        assert result.auc() > 0.99

    def test_random_attack(self):
        """Test with random attack."""
        scores = np.arange(100)
        result = AuditResult(scores, scores)
        assert 0.45 < result.auc() < 0.55

    def test_negative_scores(self):
        """Test that negative scores work correctly."""
        result = AuditResult(np.arange(-50, 0), np.arange(-100, -50))
        assert result.auc() > 0.99


class TestBetaAt:
    """Tests for beta_at method (Type-II error = 1 - TPR)."""

    def test_perfect_classifier(self):
        """Test with perfectly separated scores (low beta = strong attack)."""
        result = AuditResult(np.arange(50, 100), np.arange(0, 50))
        assert result.beta_at(alpha=0.1) < 0.1

    def test_random_classifier(self):
        """Test with identical distributions (high beta = weak attack)."""
        scores = np.arange(100)
        result = AuditResult(scores, scores)
        beta = result.beta_at(alpha=0.1)
        assert 0.8 < beta < 0.95

    def test_multiple_alphas(self):
        """Test with multiple alpha values."""
        result = AuditResult(np.arange(50, 100), np.arange(0, 50))

        alphas = np.array([0.01, 0.05, 0.1])
        betas = result.beta_at(alpha=alphas)

        assert len(betas) == 3
        # beta decreases as alpha increases (more FP allowed → fewer FN)
        assert np.all(betas[:-1] >= betas[1:])

    def test_invalid_alpha(self):
        """Test that invalid alpha raises ValueError."""
        result = AuditResult([1, 2], [3, 4])
        with pytest.raises(ValueError, match="fpr must be in"):
            result.beta_at(alpha=-0.1)
        with pytest.raises(ValueError, match="fpr must be in"):
            result.beta_at(alpha=1.5)


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
        eps = result.epsilon_one_run(significance=0.05, delta=0, threshold=5)
        assert eps >= 0

    def test_large_separation(self):
        """Test with very large score separation."""
        result = AuditResult(np.arange(1000, 2000), np.arange(0, 1000))

        eps = result.epsilon_one_run(significance=0.05, delta=0, threshold=1000)
        assert eps > 5.0


class TestAucCI:
    """Tests for auc() confidence interval support."""

    def test_basic_ci(self):
        """Test basic auc CI functionality."""
        rng = np.random.default_rng(42)
        result = AuditResult(rng.normal(2.0, 1.0, 100), rng.normal(0.0, 1.0, 100))

        ci = result.auc(confidence=0.95, num_samples=50, key=key(42))

        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] < ci[1]

    def test_ci_reproducibility(self):
        """Test that auc CI is reproducible with key."""
        result = AuditResult(np.arange(50, 100), np.arange(0, 50))

        ci1 = result.auc(confidence=0.95, num_samples=20, key=key(42))
        ci2 = result.auc(confidence=0.95, num_samples=20, key=key(42))

        assert ci1 == ci2

    def test_point_estimate_unchanged(self):
        """Test that auc() without confidence returns a float."""
        result = AuditResult(np.arange(50, 100), np.arange(0, 50))

        val = result.auc()
        assert isinstance(val, float)

    def test_ci_contains_point_estimate(self):
        """Test that CI contains the point estimate."""
        rng = np.random.default_rng(42)
        result = AuditResult(rng.normal(2.0, 1.0, 200), rng.normal(0.0, 1.0, 200))

        point = result.auc()
        ci = result.auc(confidence=0.95, num_samples=200, key=key(42))

        assert ci[0] <= point <= ci[1]

    def test_invalid_confidence(self):
        """Test that invalid confidence raises ValueError."""
        result = AuditResult(np.arange(50, 100), np.arange(0, 50))

        with pytest.raises(ValueError):
            result.auc(confidence=0.0)
        with pytest.raises(ValueError):
            result.auc(confidence=1.0)
        with pytest.raises(ValueError):
            result.auc(confidence=-0.1)


class TestCoinFlip:
    """Tests for CoinFlip (partitioning only)."""

    def test_basic_construction(self):
        """Test constructing CoinFlip with canary indices."""
        canary_idx = np.arange(100)
        cf = CoinFlip(canary_idx, key=key(42))

        assert cf.num_canaries == 100
        assert len(cf.in_indices) + len(cf.out_indices) == 100
        assert len(cf.in_indices) > 0
        assert len(cf.out_indices) > 0

    def test_coin_flip_reproducibility(self):
        """Test that same seed gives same coin flips."""
        canary_idx = np.arange(200)
        cf1 = CoinFlip(canary_idx, key=key(42))
        cf2 = CoinFlip(canary_idx, key=key(42))

        np.testing.assert_array_equal(cf1.in_indices, cf2.in_indices)
        np.testing.assert_array_equal(cf1.out_indices, cf2.out_indices)

    def test_different_seeds_give_different_splits(self):
        """Test that different seeds give different splits."""
        canary_idx = np.arange(200)
        cf1 = CoinFlip(canary_idx, key=key(42))
        cf2 = CoinFlip(canary_idx, key=key(99))

        assert not np.array_equal(cf1.in_indices, cf2.in_indices)

    def test_indices_are_subset_of_canaries(self):
        """Test that in/out indices are subsets of canary indices."""
        canary_idx = np.array([10, 20, 30, 40, 50])
        cf = CoinFlip(canary_idx, key=key(42))

        for idx in cf.in_indices:
            assert idx in canary_idx
        for idx in cf.out_indices:
            assert idx in canary_idx

    def test_no_overlap(self):
        """Test that in and out indices don't overlap."""
        canary_idx = np.arange(100)
        cf = CoinFlip(canary_idx, key=key(42))

        in_set = set(cf.in_indices.tolist())
        out_set = set(cf.out_indices.tolist())
        assert len(in_set & out_set) == 0

    def test_train_indices(self):
        """Test train_indices excludes out canaries and returns list."""
        canary_idx = np.array([5, 15, 25])
        cf = CoinFlip(canary_idx, key=key(42))

        train_idx = cf.train_indices(dataset_size=30)
        assert isinstance(train_idx, list)
        train_set = set(train_idx)

        for idx in cf.out_indices:
            assert idx not in train_set
        for idx in cf.in_indices:
            assert idx in train_set

        non_canary = set(range(30)) - set(canary_idx.tolist())
        for idx in non_canary:
            assert idx in train_set

    def test_split_scores(self):
        """Test split_scores returns correct in/out arrays."""
        canary_idx = np.arange(100)
        cf = CoinFlip(canary_idx, key=key(42))

        scores = np.zeros(100)
        scores[cf._in_mask] = 10.0
        scores[~cf._in_mask] = 0.0

        in_scores, out_scores = cf.split_scores(scores)

        assert len(in_scores) == len(cf.in_indices)
        assert len(out_scores) == len(cf.out_indices)
        np.testing.assert_array_equal(in_scores, 10.0)
        np.testing.assert_array_equal(out_scores, 0.0)

    def test_split_scores_wrong_length_raises(self):
        """Test that wrong-length scores raise ValueError."""
        canary_idx = np.arange(100)
        cf = CoinFlip(canary_idx, key=key(42))

        with pytest.raises(ValueError, match="Expected 100 scores"):
            cf.split_scores(np.zeros(50))

    def test_empty_canaries_raises(self):
        """Test that empty canary indices raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            CoinFlip(np.array([]), key=key(42))

    def test_repr(self):
        """Test CoinFlip repr."""
        cf = CoinFlip(np.arange(100), key=key(42))
        r = repr(cf)
        assert "CoinFlip" in r
        assert "num_canaries=100" in r
        assert "n_in=" in r
        assert "n_out=" in r


class TestOneRunEstimator:
    """Tests for OneRunEstimator (wraps CoinFlip + estimation)."""

    def test_audit_produces_audit_result(self):
        """Test that audit() returns correct AuditResult."""
        canary_idx = np.arange(100)
        cf = CoinFlip(canary_idx, key=key(42))
        estimator = OneRunEstimator(cf, dataset=list(range(200)))

        scores = np.zeros(100)
        scores[cf._in_mask] = 10.0
        scores[~cf._in_mask] = 0.0

        result = estimator.audit(scores)

        assert isinstance(result, AuditResult)
        assert result.n_in == len(cf.in_indices)
        assert result.n_out == len(cf.out_indices)
        assert result.auc() > 0.99

    def test_end_to_end_one_run_audit(self):
        """Test complete one-run workflow with simulated scores."""
        rng = np.random.default_rng(42)

        canary_idx = rng.choice(10000, size=500, replace=False)
        cf = CoinFlip(canary_idx, key=key(42))
        estimator = OneRunEstimator(cf, dataset=list(range(10000)))

        scores = np.empty(500)
        scores[cf._in_mask] = rng.normal(loc=0.7, scale=0.3, size=cf._in_mask.sum())
        scores[~cf._in_mask] = rng.normal(loc=0.3, scale=0.3, size=(~cf._in_mask).sum())

        result = estimator.audit(scores)
        assert result.auc() > 0.6
        assert result.epsilon_one_run(significance=0.05, delta=1e-5) > 0

    def test_repr(self):
        """Test OneRunEstimator repr."""
        cf = CoinFlip(np.arange(100), key=key(42))
        estimator = OneRunEstimator(cf, dataset=list(range(200)))
        r = repr(estimator)
        assert "OneRunEstimator" in r
        assert "num_canaries=100" in r
        assert "n_in=" in r
        assert "n_out=" in r


class TestAuditResultRepr:
    """Tests for AuditResult __repr__ and summary."""

    def test_repr(self):
        """Test __repr__ contains key info."""
        result = AuditResult(np.arange(50, 100), np.arange(0, 50))
        r = repr(result)
        assert "AuditResult" in r
        assert "n_in=50" in r
        assert "n_out=50" in r
        assert "auc=" in r

    def test_summary(self):
        """Test summary() produces multi-line report."""
        result = AuditResult(np.arange(50, 100), np.arange(0, 50))
        s = result.summary()
        assert "Audit Summary" in s
        assert "Samples:" in s
        assert "AUC:" in s
        assert "one-run" in s
        assert "β @" in s
        assert "Max accuracy" in s

    def test_summary_custom_params(self):
        """Test summary with custom significance and delta."""
        result = AuditResult(np.arange(50, 100), np.arange(0, 50))
        s = result.summary(significance=0.01, delta=1e-5)
        assert "\u03b1=0.01" in s
        assert "\u03b4=1e-05" in s

    def test_summary_theoretical_epsilon(self):
        """Test summary includes theoretical epsilon when provided."""
        result = AuditResult(np.arange(50, 100), np.arange(0, 50))
        s = result.summary(theoretical_epsilon=3.0)
        assert "theoretical" in s
        assert "3.0000" in s

    def test_summary_without_theoretical_epsilon(self):
        """Test summary excludes theoretical epsilon when not provided."""
        result = AuditResult(np.arange(50, 100), np.arange(0, 50))
        s = result.summary()
        assert "theoretical" not in s


class TestEpsilonAt:
    """Tests for epsilon_at method."""

    def test_defaults_to_one_run(self):
        """Test that epsilon_at uses one_run method."""
        result = AuditResult(np.arange(50, 100), np.arange(0, 50))
        eps_at = result.epsilon_at(delta=0.0)
        eps_or = result.epsilon_one_run(significance=0.05, delta=0.0)
        assert eps_at == eps_or

    def test_delta_passthrough(self):
        """Test that delta is passed through correctly."""
        result = AuditResult(np.arange(50, 100), np.arange(0, 50))
        eps_0 = result.epsilon_at(delta=0.0)
        eps_d = result.epsilon_at(delta=0.1)
        # With delta > 0, epsilon should generally be different
        assert isinstance(eps_0, float)
        assert isinstance(eps_d, float)


class TestCoinFlipFunction:
    """Tests for auditing.coin_flip() module-level function."""

    def test_basic_coin_flip(self):
        """Test coin_flip creates CoinFlip from dataset."""
        dataset = list(range(1000))
        cf = auditing.coin_flip(dataset, num_canaries=100, key=key(42))

        assert isinstance(cf, CoinFlip)
        assert cf.num_canaries == 100
        assert len(cf.in_indices) + len(cf.out_indices) == 100

    def test_coin_flip_too_many_canaries(self):
        """Test that requesting more canaries than dataset size raises."""
        dataset = list(range(10))
        with pytest.raises(ValueError, match="exceeds dataset size"):
            auditing.coin_flip(dataset, num_canaries=20, key=key(42))


class TestOneRunFunction:
    """Tests for auditing.one_run() module-level function."""

    def test_basic_one_run(self):
        """Test one_run creates OneRunEstimator from CoinFlip."""
        dataset = list(range(1000))
        cf = auditing.coin_flip(dataset, num_canaries=100, key=key(42))
        estimator = auditing.one_run(cf, dataset=dataset, batch_argnums=(1,))

        assert isinstance(estimator, OneRunEstimator)
        assert estimator.coin_flip is cf
        assert estimator._batch_argnums == (1,)

    def test_one_run_train_indices(self):
        """Test that one_run result has correct train_indices."""
        dataset = list(range(100))
        cf = auditing.coin_flip(dataset, num_canaries=10, key=key(42))
        estimator = auditing.one_run(cf, dataset=dataset)

        train_set = set(estimator.train_indices)
        for idx in cf.out_indices:
            assert idx not in train_set
        for idx in cf.in_indices:
            assert idx in train_set


class TestSetup:
    """Tests for auditing.setup() convenience function."""

    def test_basic_setup(self):
        """Test setup creates OneRunEstimator from dataset."""
        dataset = list(range(1000))
        audit_state = auditing.setup(dataset, num_canaries=100, key=key(42))

        assert isinstance(audit_state, OneRunEstimator)
        cf = audit_state.coin_flip
        assert cf.num_canaries == 100
        assert len(cf.in_indices) + len(cf.out_indices) == 100

    def test_setup_reproducibility(self):
        """Test setup is reproducible with same seed."""
        dataset = list(range(1000))
        s1 = auditing.setup(dataset, num_canaries=100, key=key(42))
        s2 = auditing.setup(dataset, num_canaries=100, key=key(42))

        np.testing.assert_array_equal(
            s1.coin_flip.canary_indices, s2.coin_flip.canary_indices
        )
        np.testing.assert_array_equal(s1.coin_flip.in_indices, s2.coin_flip.in_indices)

    def test_setup_different_seeds(self):
        """Test different seeds give different partitions."""
        dataset = list(range(1000))
        s1 = auditing.setup(dataset, num_canaries=100, key=key(42))
        s2 = auditing.setup(dataset, num_canaries=100, key=key(99))

        assert not np.array_equal(
            s1.coin_flip.canary_indices, s2.coin_flip.canary_indices
        )

    def test_setup_too_many_canaries(self):
        """Test that requesting more canaries than dataset size raises."""
        dataset = list(range(10))
        with pytest.raises(ValueError, match="exceeds dataset size"):
            auditing.setup(dataset, num_canaries=20, key=key(42))

    def test_setup_canaries_are_valid_indices(self):
        """Test all canary indices are valid for the dataset."""
        dataset = list(range(500))
        audit_state = auditing.setup(dataset, num_canaries=100, key=key(42))

        assert np.all(audit_state.coin_flip.canary_indices >= 0)
        assert np.all(audit_state.coin_flip.canary_indices < 500)

    def test_train_indices(self):
        """Test that train_indices excludes out-canaries."""
        dataset = list(range(100))
        audit_state = auditing.setup(dataset, num_canaries=10, key=key(42))

        cf = audit_state.coin_flip
        train_set = set(audit_state.train_indices)

        for idx in cf.out_indices:
            assert idx not in train_set
        for idx in cf.in_indices:
            assert idx in train_set

    def test_repr(self):
        """Test OneRunEstimator repr from setup."""
        audit_state = auditing.setup(list(range(1000)), num_canaries=100, key=key(42))
        r = repr(audit_state)
        assert "OneRunEstimator" in r
        assert "num_canaries=100" in r
        assert "n_in=" in r
        assert "n_out=" in r
