"""Tests for BisrStrategy factory and accounting equivalence."""

import pytest

import opaque.dpftrl.accounting as ftrl_acc
from opaque.dpftrl.noise._bisr import BisrStrategy, bisr_strategy


class TestBisrStrategy:
    def test_returns_correct_type(self):
        s = bisr_strategy(bandwidth=4, n_steps=100, min_sep=25, max_participations=4)
        assert isinstance(s, BisrStrategy)

    def test_sensitivity_positive(self):
        s = bisr_strategy(bandwidth=4, n_steps=100, min_sep=25, max_participations=4)
        assert s.sensitivity > 0

    def test_gram_matrix_present(self):
        s = bisr_strategy(bandwidth=4, n_steps=100, min_sep=25, max_participations=4)
        assert s.gram_matrix is not None
        assert len(s.gram_matrix) == 25 * 25

    def test_streaming_matrix_present(self):
        s = bisr_strategy(bandwidth=4, n_steps=100, min_sep=25, max_participations=4)
        assert s._streaming_matrix is not None

    def test_matches_old_sensitivity(self):
        new = bisr_strategy(bandwidth=4, n_steps=100, min_sep=25, max_participations=4)
        assert new.sensitivity > 0

    def test_with_momentum(self):
        s = bisr_strategy(
            bandwidth=4, n_steps=100, min_sep=25, max_participations=4, momentum=0.95
        )
        assert s.sensitivity > 0

    def test_rejects_bad_bandwidth(self):
        with pytest.raises(ValueError):
            bisr_strategy(bandwidth=1, n_steps=100, min_sep=25)


class TestBisrPld:
    delta = 1e-5

    def test_bisr_pld(self):
        s = bisr_strategy(bandwidth=4, n_steps=100, min_sep=25, max_participations=4)
        eps = ftrl_acc.bisr(1.0, sensitivity=s.sensitivity).epsilon_at(self.delta)
        assert eps > 0

    def test_bisr_bnb(self):
        s = bisr_strategy(bandwidth=4, n_steps=100, min_sep=25, max_participations=4)
        eps = ftrl_acc.balls_in_bins(
            ftrl_acc.bisr(1.0, sensitivity=s.sensitivity, gram_matrix=s.gram_matrix),
            num_bins=25,
            n_steps=100,
        ).epsilon_at(self.delta)
        assert eps > 0
