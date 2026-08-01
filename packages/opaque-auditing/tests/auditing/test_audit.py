"""Tests for OneRunEstimate, CoinFlip, one_run, and coin_flip."""

import numpy as np
import pytest

import opaque.auditing as auditing
from opaque.auditing import one_run
from opaque.auditing.types import CoinFlip, OneRunEstimate
from opaque.random import key
from opaque.random.types import RngKey


def _flip(canary_indices: np.ndarray, *, key: RngKey) -> CoinFlip:
    """Test helper: coin-flip partition from raw indices + RNG key."""
    canary_indices = np.asarray(canary_indices)
    if canary_indices.ndim != 1 or canary_indices.size == 0:
        raise ValueError("canary_indices must be a non-empty 1-D array")
    rng = np.random.default_rng(key.seed)
    in_mask = rng.random(len(canary_indices)) < 0.5
    return CoinFlip(
        num_canaries=len(canary_indices),
        canary_indices=canary_indices,
        _in_mask=in_mask,
        in_indices=canary_indices[in_mask],
        out_indices=canary_indices[~in_mask],
    )


def _make_estimate(in_scores, out_scores):
    """Helper: build a OneRunEstimate from raw in/out score arrays."""
    n_in = len(in_scores)
    n_out = len(out_scores)
    canary_indices = np.arange(n_in + n_out)
    mask = np.array([True] * n_in + [False] * n_out)
    cf = CoinFlip(
        num_canaries=n_in + n_out,
        canary_indices=canary_indices,
        _in_mask=mask,
        in_indices=canary_indices[mask],
        out_indices=canary_indices[~mask],
    )
    scores = np.empty(n_in + n_out)
    scores[mask] = in_scores
    scores[~mask] = out_scores
    return one_run(scores, coin_flip=cf)


class TestConstruction:
    """Tests for OneRunEstimate construction via one_run()."""

    def test_basic_construction(self):
        estimate = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        assert estimate.n_in == 50
        assert estimate.n_out == 50

    def test_list_input(self):
        estimate = _make_estimate([1, 2, 3], [4, 5, 6])
        assert estimate.n_in == 3

    def test_empty_in_scores(self):
        """Test that empty in_scores raises ValueError."""
        canary_indices = np.arange(3)
        mask = np.array([False, False, False])
        cf = CoinFlip(
            num_canaries=3,
            canary_indices=canary_indices,
            _in_mask=mask,
            in_indices=canary_indices[mask],
            out_indices=canary_indices[~mask],
        )
        with pytest.raises(ValueError, match="non-empty"):
            one_run(np.array([1.0, 2.0, 3.0]), coin_flip=cf)

    def test_empty_out_scores(self):
        """Test that empty out_scores raises ValueError."""
        canary_indices = np.arange(3)
        mask = np.array([True, True, True])
        cf = CoinFlip(
            num_canaries=3,
            canary_indices=canary_indices,
            _in_mask=mask,
            in_indices=canary_indices[mask],
            out_indices=canary_indices[~mask],
        )
        with pytest.raises(ValueError, match="non-empty"):
            one_run(np.array([1.0, 2.0, 3.0]), coin_flip=cf)

    @pytest.mark.parametrize("invalid_score", [np.nan, np.inf, -np.inf])
    @pytest.mark.parametrize("partition", ["in", "out"])
    def test_non_finite_score_raises(self, invalid_score, partition):
        in_scores = np.array([1.0, 2.0, 3.0])
        out_scores = np.array([4.0, 5.0, 6.0])
        scores = in_scores if partition == "in" else out_scores
        scores[1] = invalid_score

        with pytest.raises(ValueError, match="scores must contain only finite values"):
            _make_estimate(in_scores, out_scores)


class TestEpsilonAt:
    """Tests for eps_delta().epsilon_at()."""

    def test_perfect_separation(self):
        estimate = _make_estimate(list(range(100, 150)), list(range(50)))
        eps = estimate.eps_delta().epsilon_at(significance=0.05, delta=0, threshold=75)
        assert eps > 0

    def test_no_separation(self):
        scores = list(range(100))
        estimate = _make_estimate(scores, scores)
        eps = estimate.eps_delta().epsilon_at(significance=0.05, delta=0)
        assert eps < 1.0

    def test_invalid_significance(self):
        estimate = _make_estimate([1, 2], [3, 4])
        with pytest.raises(ValueError, match="significance must be in"):
            estimate.eps_delta().epsilon_at(significance=0.0)
        with pytest.raises(ValueError, match="significance must be in"):
            estimate.eps_delta().epsilon_at(significance=0.6)

    def test_invalid_delta(self):
        estimate = _make_estimate([1, 2], [3, 4])
        with pytest.raises(ValueError, match="delta must be in"):
            estimate.eps_delta().epsilon_at(significance=0.05, delta=-0.1)
        with pytest.raises(ValueError, match="delta must be in"):
            estimate.eps_delta().epsilon_at(significance=0.05, delta=1.5)

    def test_delta_passthrough(self):
        estimate = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        eps_0 = estimate.eps_delta().epsilon_at(delta=0.0)
        eps_d = estimate.eps_delta().epsilon_at(delta=0.1)
        assert isinstance(eps_0, float)
        assert isinstance(eps_d, float)


class TestPldMirrorDispatch:
    """OneRunEstimate's Pld-mirror surface dispatches to gdp() by default."""

    def test_epsilon_at_matches_gdp(self):
        estimate = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        assert estimate.epsilon_at(delta=1e-5) == estimate.gdp().epsilon_at(delta=1e-5)

    def test_delta_at_matches_gdp(self):
        estimate = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        assert estimate.delta_at(epsilon=2.0) == estimate.gdp().delta_at(epsilon=2.0)

    def test_beta_at_matches_gdp(self):
        estimate = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        assert estimate.beta_at(alpha=0.1) == estimate.gdp().beta_at(alpha=0.1)

    def test_advantage_matches_gdp(self):
        estimate = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        assert estimate.advantage() == estimate.gdp().advantage()

    def test_epsilon_at_rejects_delta_zero(self):
        """gdp dispatch requires δ > 0."""
        estimate = _make_estimate([1, 2], [3, 4])
        with pytest.raises(ValueError, match="delta > 0"):
            estimate.epsilon_at(delta=0.0)


class TestAttackAuc:
    """Tests for attack_auc method."""

    def test_perfect_attack(self):
        estimate = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        assert estimate.attack_auc() > 0.99

    def test_random_attack(self):
        scores = np.arange(100)
        estimate = _make_estimate(scores, scores)
        assert 0.45 < estimate.attack_auc() < 0.55

    def test_negative_scores(self):
        estimate = _make_estimate(np.arange(-50, 0), np.arange(-100, -50))
        assert estimate.attack_auc() > 0.99


class TestAttackBetaAt:
    """Tests for attack_beta_at method (empirical Type-II error = 1 - TPR)."""

    def test_perfect_classifier(self):
        estimate = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        assert estimate.attack_beta_at(alpha=0.1) < 0.1

    def test_random_classifier(self):
        scores = np.arange(100)
        estimate = _make_estimate(scores, scores)
        beta = estimate.attack_beta_at(alpha=0.1)
        assert 0.8 < beta < 0.95

    def test_multiple_alphas(self):
        estimate = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        alphas = np.array([0.01, 0.05, 0.1])
        betas = estimate.attack_beta_at(alpha=alphas)
        assert len(betas) == 3
        assert np.all(betas[:-1] >= betas[1:])

    def test_invalid_alpha(self):
        estimate = _make_estimate([1, 2], [3, 4])
        with pytest.raises(ValueError, match="fpr must be in"):
            estimate.attack_beta_at(alpha=-0.1)
        with pytest.raises(ValueError, match="fpr must be in"):
            estimate.attack_beta_at(alpha=1.5)


class TestEdgeCases:
    """Edge case tests."""

    def test_single_score_each(self):
        estimate = _make_estimate([10], [0])
        eps = estimate.eps_delta().epsilon_at(significance=0.05, delta=0, threshold=5)
        assert eps >= 0

    def test_large_separation(self):
        estimate = _make_estimate(np.arange(1000, 2000), np.arange(0, 1000))
        eps = estimate.eps_delta().epsilon_at(
            significance=0.05, delta=0, threshold=1000
        )
        assert eps > 5.0


class TestAucCI:
    """Tests for auc() confidence interval support."""

    def test_basic_ci(self):
        rng = np.random.default_rng(42)
        estimate = _make_estimate(rng.normal(2.0, 1.0, 100), rng.normal(0.0, 1.0, 100))
        ci = estimate.attack_auc(confidence=0.95, num_samples=50, key=key(42))
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] < ci[1]

    def test_ci_reproducibility(self):
        estimate = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        ci1 = estimate.attack_auc(confidence=0.95, num_samples=20, key=key(42))
        ci2 = estimate.attack_auc(confidence=0.95, num_samples=20, key=key(42))
        assert ci1 == ci2

    def test_point_estimate_unchanged(self):
        estimate = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        val = estimate.attack_auc()
        assert isinstance(val, float)

    def test_ci_contains_point_estimate(self):
        rng = np.random.default_rng(42)
        estimate = _make_estimate(rng.normal(2.0, 1.0, 200), rng.normal(0.0, 1.0, 200))
        point = estimate.attack_auc()
        ci = estimate.attack_auc(confidence=0.95, num_samples=200, key=key(42))
        assert ci[0] <= point <= ci[1]

    def test_invalid_confidence(self):
        estimate = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        with pytest.raises(ValueError, match="confidence must be in"):
            estimate.attack_auc(confidence=0.0)
        with pytest.raises(ValueError, match="confidence must be in"):
            estimate.attack_auc(confidence=1.0)
        with pytest.raises(ValueError, match="confidence must be in"):
            estimate.attack_auc(confidence=-0.1)


class TestCoinFlip:
    """Tests for CoinFlip (partitioning only)."""

    def test_basic_construction(self):
        canary_idx = np.arange(100)
        cf = _flip(canary_idx, key=key(42))
        assert cf.num_canaries == 100
        assert len(cf.in_indices) + len(cf.out_indices) == 100
        assert len(cf.in_indices) > 0
        assert len(cf.out_indices) > 0

    def test_coin_flip_reproducibility(self):
        canary_idx = np.arange(200)
        cf1 = _flip(canary_idx, key=key(42))
        cf2 = _flip(canary_idx, key=key(42))
        np.testing.assert_array_equal(cf1.in_indices, cf2.in_indices)
        np.testing.assert_array_equal(cf1.out_indices, cf2.out_indices)

    def test_different_seeds_give_different_splits(self):
        canary_idx = np.arange(200)
        cf1 = _flip(canary_idx, key=key(42))
        cf2 = _flip(canary_idx, key=key(99))
        assert not np.array_equal(cf1.in_indices, cf2.in_indices)

    def test_indices_are_subset_of_canaries(self):
        canary_idx = np.array([10, 20, 30, 40, 50])
        cf = _flip(canary_idx, key=key(42))
        for idx in cf.in_indices:
            assert idx in canary_idx
        for idx in cf.out_indices:
            assert idx in canary_idx

    def test_no_overlap(self):
        canary_idx = np.arange(100)
        cf = _flip(canary_idx, key=key(42))
        in_set = set(cf.in_indices.tolist())
        out_set = set(cf.out_indices.tolist())
        assert len(in_set & out_set) == 0

    def test_train_indices(self):
        canary_idx = np.array([5, 15, 25])
        cf = _flip(canary_idx, key=key(42))
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
        canary_idx = np.arange(100)
        cf = _flip(canary_idx, key=key(42))
        scores = np.zeros(100)
        scores[cf._in_mask] = 10.0
        scores[~cf._in_mask] = 0.0
        in_scores, out_scores = cf.split_scores(scores)
        assert len(in_scores) == len(cf.in_indices)
        assert len(out_scores) == len(cf.out_indices)
        np.testing.assert_array_equal(in_scores, 10.0)
        np.testing.assert_array_equal(out_scores, 0.0)

    def test_split_scores_wrong_length_raises(self):
        canary_idx = np.arange(100)
        cf = _flip(canary_idx, key=key(42))
        with pytest.raises(ValueError, match="Expected 100 scores"):
            cf.split_scores(np.zeros(50))

    def test_empty_canaries_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _flip(np.array([]), key=key(42))

    def test_repr(self):
        cf = _flip(np.arange(100), key=key(42))
        r = repr(cf)
        assert "CoinFlip" in r
        assert "num_canaries=100" in r
        assert "n_in=" in r
        assert "n_out=" in r


class TestOneRunFunction:
    """Tests for one_run() free function."""

    def test_one_run_produces_estimate(self):
        canary_idx = np.arange(100)
        cf = _flip(canary_idx, key=key(42))
        scores = np.zeros(100)
        scores[cf._in_mask] = 10.0
        scores[~cf._in_mask] = 0.0

        estimate = one_run(scores, coin_flip=cf)

        assert isinstance(estimate, OneRunEstimate)
        assert estimate.n_in == len(cf.in_indices)
        assert estimate.n_out == len(cf.out_indices)
        assert estimate.attack_auc() > 0.99

    def test_end_to_end_one_run(self):
        rng = np.random.default_rng(42)
        canary_idx = rng.choice(10000, size=500, replace=False)
        cf = _flip(canary_idx, key=key(42))

        scores = np.empty(500)
        scores[cf._in_mask] = rng.normal(loc=0.7, scale=0.3, size=cf._in_mask.sum())
        scores[~cf._in_mask] = rng.normal(loc=0.3, scale=0.3, size=(~cf._in_mask).sum())

        estimate = one_run(scores, coin_flip=cf)
        assert estimate.attack_auc() > 0.6
        assert estimate.eps_delta().epsilon_at(significance=0.05, delta=1e-5) > 0


class TestOneRunEstimateRepr:
    """Tests for OneRunEstimate __repr__."""

    def test_repr(self):
        estimate = _make_estimate(np.arange(50, 100), np.arange(0, 50))
        r = repr(estimate)
        assert "OneRunEstimate" in r
        assert "n_in=50" in r
        assert "n_out=50" in r
        assert "auc=" in r


class TestCoinFlipFunction:
    """Tests for auditing.coin_flip() module-level function."""

    def test_basic_coin_flip(self):
        dataset = list(range(1000))
        cf = auditing.coin_flip(dataset, num_canaries=100, key=key(42))
        assert isinstance(cf, CoinFlip)
        assert cf.num_canaries == 100
        assert len(cf.in_indices) + len(cf.out_indices) == 100

    def test_coin_flip_too_many_canaries(self):
        dataset = list(range(10))
        with pytest.raises(ValueError, match="exceeds dataset size"):
            auditing.coin_flip(dataset, num_canaries=20, key=key(42))
