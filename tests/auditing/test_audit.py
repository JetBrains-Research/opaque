"""Tests for AuditResult and CoinFlipExperiment classes."""

import numpy as np
import pytest

import opaque.auditing as auditing
from opaque.auditing import AuditResult, CoinFlipExperiment


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

        eps = result.epsilon_clopper_pearson(significance=0.05, delta=0, threshold=75)
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

        eps = result.epsilon_clopper_pearson(significance=0.05, delta=0, threshold=50)
        assert eps > 0


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
        eps = result.epsilon_clopper_pearson(significance=0.05, delta=0, threshold=5)
        assert eps >= 0

    def test_large_separation(self):
        """Test with very large score separation."""
        result = AuditResult(np.arange(1000, 2000), np.arange(0, 1000))

        eps = result.epsilon_clopper_pearson(significance=0.05, delta=0, threshold=1000)
        assert eps > 5.0


class TestBootstrap:
    """Tests for bootstrap method."""

    def test_basic_bootstrap(self):
        """Test basic bootstrap functionality."""
        from opaque.auditing import BootstrapParams

        rng = np.random.default_rng(42)
        result = AuditResult(rng.normal(2.0, 1.0, 100), rng.normal(0.0, 1.0, 100))

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
        result = AuditResult(rng.normal(2.0, 1.0, 100), rng.normal(0.0, 1.0, 100))

        params = BootstrapParams(num_samples=50, quantiles=(0.1, 0.5, 0.9), seed=42)
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


class TestCoinFlipExperiment:
    """Tests for CoinFlipExperiment (one-run auditing setup)."""

    def test_basic_construction(self):
        """Test constructing experiment with canary indices."""
        canary_idx = np.arange(100)
        exp = CoinFlipExperiment(canary_idx, seed=42)

        assert exp.num_canaries == 100
        assert len(exp.in_indices) + len(exp.out_indices) == 100
        # With 100 coins, both groups should be non-empty
        assert len(exp.in_indices) > 0
        assert len(exp.out_indices) > 0

    def test_coin_flip_reproducibility(self):
        """Test that same seed gives same coin flips."""
        canary_idx = np.arange(200)
        exp1 = CoinFlipExperiment(canary_idx, seed=42)
        exp2 = CoinFlipExperiment(canary_idx, seed=42)

        np.testing.assert_array_equal(exp1.in_indices, exp2.in_indices)
        np.testing.assert_array_equal(exp1.out_indices, exp2.out_indices)

    def test_different_seeds_give_different_splits(self):
        """Test that different seeds give different splits."""
        canary_idx = np.arange(200)
        exp1 = CoinFlipExperiment(canary_idx, seed=42)
        exp2 = CoinFlipExperiment(canary_idx, seed=99)

        # Extremely unlikely to be identical with different seeds
        assert not np.array_equal(exp1.in_indices, exp2.in_indices)

    def test_indices_are_subset_of_canaries(self):
        """Test that in/out indices are subsets of canary indices."""
        canary_idx = np.array([10, 20, 30, 40, 50])
        exp = CoinFlipExperiment(canary_idx, seed=42)

        for idx in exp.in_indices:
            assert idx in canary_idx
        for idx in exp.out_indices:
            assert idx in canary_idx

    def test_no_overlap(self):
        """Test that in and out indices don't overlap."""
        canary_idx = np.arange(100)
        exp = CoinFlipExperiment(canary_idx, seed=42)

        in_set = set(exp.in_indices.tolist())
        out_set = set(exp.out_indices.tolist())
        assert len(in_set & out_set) == 0

    def test_train_indices(self):
        """Test train_indices excludes out canaries."""
        canary_idx = np.array([5, 15, 25])
        exp = CoinFlipExperiment(canary_idx, seed=42)

        train_idx = exp.train_indices(dataset_size=30)
        train_set = set(train_idx.tolist())

        # Out canaries must not be in training set
        for idx in exp.out_indices:
            assert idx not in train_set

        # In canaries must be in training set
        for idx in exp.in_indices:
            assert idx in train_set

        # Non-canary indices must be in training set
        non_canary = set(range(30)) - set(canary_idx.tolist())
        for idx in non_canary:
            assert idx in train_set

    def test_audit_produces_audit_result(self):
        """Test that audit() returns correct AuditResult."""
        canary_idx = np.arange(100)
        exp = CoinFlipExperiment(canary_idx, seed=42)

        # Simulate: in-canaries get high scores, out-canaries get low
        scores = np.zeros(100)
        scores[exp._in_mask] = 10.0
        scores[~exp._in_mask] = 0.0

        result = exp.audit(scores)

        assert isinstance(result, AuditResult)
        assert result.n_in == len(exp.in_indices)
        assert result.n_out == len(exp.out_indices)
        # Perfect separation → high AUROC
        assert result.auroc() > 0.99

    def test_audit_wrong_length_raises(self):
        """Test that wrong-length scores raise ValueError."""
        canary_idx = np.arange(100)
        exp = CoinFlipExperiment(canary_idx, seed=42)

        with pytest.raises(ValueError, match="Expected 100 scores"):
            exp.audit(np.zeros(50))

    def test_empty_canaries_raises(self):
        """Test that empty canary indices raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            CoinFlipExperiment(np.array([]))

    def test_end_to_end_one_run_audit(self):
        """Test complete one-run auditing workflow with simulated scores."""
        rng = np.random.default_rng(42)

        # Setup: 500 canaries from a 10k dataset
        canary_idx = rng.choice(10000, size=500, replace=False)
        exp = CoinFlipExperiment(canary_idx, seed=42)

        # Simulate membership scores: in-canaries score higher
        scores = np.empty(500)
        scores[exp._in_mask] = rng.normal(loc=0.7, scale=0.3, size=exp._in_mask.sum())
        scores[~exp._in_mask] = rng.normal(
            loc=0.3, scale=0.3, size=(~exp._in_mask).sum()
        )

        # Audit
        result = exp.audit(scores)
        assert result.auroc() > 0.6
        assert result.epsilon_one_run(significance=0.05, delta=1e-5) > 0

    def test_repr(self):
        """Test CoinFlipExperiment repr."""
        exp = CoinFlipExperiment(np.arange(100), seed=42)
        r = repr(exp)
        assert "CoinFlipExperiment" in r
        assert "num_canaries=100" in r
        assert "n_in=" in r
        assert "n_out=" in r

    def test_subset(self):
        """Test subset() returns a torch Subset excluding out-canaries."""
        import torch
        from torch.utils.data import TensorDataset

        dataset = TensorDataset(torch.arange(50), torch.arange(50))
        canary_idx = np.array([5, 15, 25, 35, 45])
        exp = CoinFlipExperiment(canary_idx, seed=42)

        sub = exp.subset(dataset)
        assert len(sub) == 50 - len(exp.out_indices)

        # All out-canaries excluded
        sub_indices = set(sub.indices)
        for idx in exp.out_indices:
            assert idx not in sub_indices

        # All in-canaries included
        for idx in exp.in_indices:
            assert idx in sub_indices

    def test_canary_subset(self):
        """Test canary_subset() returns only canary examples."""
        import torch
        from torch.utils.data import TensorDataset

        dataset = TensorDataset(torch.arange(50), torch.arange(50))
        canary_idx = np.array([5, 15, 25, 35, 45])
        exp = CoinFlipExperiment(canary_idx, seed=42)

        sub = exp.canary_subset(dataset)
        assert len(sub) == 5
        assert sub.indices == canary_idx.tolist()


class TestAuditResultRepr:
    """Tests for AuditResult __repr__ and summary."""

    def test_repr(self):
        """Test __repr__ contains key info."""
        result = AuditResult(np.arange(50, 100), np.arange(0, 50))
        r = repr(result)
        assert "AuditResult" in r
        assert "n_in=50" in r
        assert "n_out=50" in r
        assert "auroc=" in r

    def test_summary(self):
        """Test summary() produces multi-line report."""
        result = AuditResult(np.arange(50, 100), np.arange(0, 50))
        s = result.summary()
        assert "Audit Summary" in s
        assert "Samples:" in s
        assert "AUROC:" in s
        assert "Clopper-Pearson" in s
        assert "TPR" in s
        assert "Max accuracy" in s

    def test_summary_coin_flip_shows_one_run(self):
        """Test summary shows one-run epsilon when from coin flip."""
        exp = CoinFlipExperiment(np.arange(100), seed=42)
        scores = np.zeros(100)
        scores[exp._in_mask] = 10.0
        result = exp.audit(scores)

        s = result.summary()
        assert "one-run" in s

    def test_summary_direct_hides_one_run(self):
        """Test summary hides one-run epsilon when constructed directly."""
        result = AuditResult(np.arange(50, 100), np.arange(0, 50))
        s = result.summary()
        assert "one-run" not in s

    def test_summary_custom_params(self):
        """Test summary with custom significance and delta."""
        result = AuditResult(np.arange(50, 100), np.arange(0, 50))
        s = result.summary(significance=0.01, delta=1e-5)
        assert "\u03b1=0.01" in s
        assert "\u03b4=1e-05" in s


class TestEpsilonAt:
    """Tests for epsilon_at method."""

    def test_direct_defaults_to_clopper_pearson(self):
        """Test that directly constructed AuditResult uses clopper_pearson."""
        result = AuditResult(np.arange(50, 100), np.arange(0, 50))
        eps_at = result.epsilon_at(delta=0.0)
        eps_cp = result.epsilon_clopper_pearson(significance=0.05, delta=0.0)
        assert eps_at == eps_cp

    def test_coin_flip_defaults_to_one_run(self):
        """Test that coin-flip AuditResult uses one_run."""
        exp = CoinFlipExperiment(np.arange(100), seed=42)
        scores = np.zeros(100)
        scores[exp._in_mask] = 10.0
        result = exp.audit(scores)

        eps_at = result.epsilon_at(delta=0.0)
        eps_or = result.epsilon_one_run(significance=0.05, delta=0.0)
        assert eps_at == eps_or

    def test_explicit_method_override(self):
        """Test that method parameter overrides default."""
        result = AuditResult(np.arange(50, 100), np.arange(0, 50))
        eps_or = result.epsilon_at(delta=0.0, method="one_run")
        eps_cp = result.epsilon_at(delta=0.0, method="clopper_pearson")
        # Both should be valid numbers (may differ)
        assert eps_or > 0
        assert eps_cp > 0

    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        result = AuditResult([1, 2, 3], [4, 5, 6])
        with pytest.raises(ValueError, match="method must be"):
            result.epsilon_at(method="invalid")

    def test_delta_passthrough(self):
        """Test that delta is passed through correctly."""
        result = AuditResult(np.arange(50, 100), np.arange(0, 50))
        eps_0 = result.epsilon_at(delta=0.0)
        eps_d = result.epsilon_at(delta=0.1)
        # With delta > 0, epsilon should generally be different
        assert isinstance(eps_0, float)
        assert isinstance(eps_d, float)


class TestSetup:
    """Tests for auditing.setup() module-level function."""

    def test_basic_setup(self):
        """Test setup creates experiment from dataset."""
        dataset = list(range(1000))  # Anything with len()
        exp = auditing.setup(dataset, num_canaries=100, seed=42)

        assert isinstance(exp, CoinFlipExperiment)
        assert exp.num_canaries == 100
        assert len(exp.in_indices) + len(exp.out_indices) == 100

    def test_setup_reproducibility(self):
        """Test setup is reproducible with same seed."""
        dataset = list(range(1000))
        exp1 = auditing.setup(dataset, num_canaries=100, seed=42)
        exp2 = auditing.setup(dataset, num_canaries=100, seed=42)

        np.testing.assert_array_equal(exp1._canary_indices, exp2._canary_indices)
        np.testing.assert_array_equal(exp1.in_indices, exp2.in_indices)

    def test_setup_different_seeds(self):
        """Test different seeds give different experiments."""
        dataset = list(range(1000))
        exp1 = auditing.setup(dataset, num_canaries=100, seed=42)
        exp2 = auditing.setup(dataset, num_canaries=100, seed=99)

        assert not np.array_equal(exp1._canary_indices, exp2._canary_indices)

    def test_setup_too_many_canaries(self):
        """Test that requesting more canaries than dataset size raises."""
        dataset = list(range(10))
        with pytest.raises(ValueError, match="exceeds dataset size"):
            auditing.setup(dataset, num_canaries=20, seed=42)

    def test_setup_canaries_are_valid_indices(self):
        """Test all canary indices are valid for the dataset."""
        dataset = list(range(500))
        exp = auditing.setup(dataset, num_canaries=100, seed=42)

        assert np.all(exp._canary_indices >= 0)
        assert np.all(exp._canary_indices < 500)
