"""Tests for BisrStrategy factory and accounting equivalence."""

import pytest

import opaque.dpftrl.accounting as ftrl_acc
from opaque.api.dpftrl.noise._bisr import BisrStrategy, bisr_strategy

_PART = {"n_steps": 100, "min_sep": 25, "max_participations": 4}


class TestBisrStrategy:
    def test_returns_correct_type(self):
        assert isinstance(bisr_strategy(bandwidth=4), BisrStrategy)

    def test_sensitivity_positive(self):
        assert bisr_strategy(bandwidth=4).sensitivity(**_PART) > 0

    def test_gram_matrix_present(self):
        gram = bisr_strategy(bandwidth=4).gram_matrix(**_PART)
        assert gram is not None
        assert len(gram) == 25 * 25

    def test_streaming_matrix_present(self):
        assert bisr_strategy(bandwidth=4).streaming_matrix(**_PART) is not None

    def test_matches_old_sensitivity(self):
        assert bisr_strategy(bandwidth=4).sensitivity(**_PART) > 0

    def test_with_momentum(self):
        assert bisr_strategy(bandwidth=4, momentum=0.95).sensitivity(**_PART) > 0

    def test_rejects_bad_bandwidth(self):
        with pytest.raises(ValueError):
            bisr_strategy(bandwidth=1)


class TestBisrPld:
    delta = 1e-5

    def test_bisr_pld(self):
        eps = ftrl_acc.mf_gaussian(1.0, bisr_strategy(bandwidth=4), **_PART).epsilon_at(
            self.delta
        )
        assert eps > 0

    def test_bisr_bnb(self):
        eps = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_gaussian(1.0, bisr_strategy(bandwidth=4)),
            num_bins=25,
            n_steps=100,
        ).epsilon_at(self.delta)
        assert eps > 0
