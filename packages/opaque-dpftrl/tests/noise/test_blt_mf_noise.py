"""Tests for BltStrategy factory and accounting equivalence."""

import opaque.dpftrl.accounting as ftrl_acc
from opaque.api.dpftrl.noise._blt import BltStrategy, blt_strategy


_PART = dict(n_steps=100, min_sep=25, max_participations=4)


class TestBltStrategy:
    def test_returns_correct_type(self):
        assert isinstance(blt_strategy(momentum=0.95), BltStrategy)

    def test_sensitivity_positive(self):
        s = blt_strategy(momentum=0.95)
        assert s.sensitivity(**_PART) > 0

    def test_gram_matrix_present(self):
        s = blt_strategy(momentum=0.95)
        gram = s.gram_matrix(**_PART)
        assert gram is not None
        assert len(gram) == 25 * 25

    def test_coefficients_length(self):
        s = blt_strategy(momentum=0.95)
        assert s.coefficients(**_PART).shape[0] == 100

    def test_streaming_matrix_present(self):
        s = blt_strategy(momentum=0.95)
        assert s.streaming_matrix(**_PART) is not None

    def test_matches_old_sensitivity(self):
        assert blt_strategy(momentum=0.95).sensitivity(**_PART) > 0

    def test_single_participation(self):
        s = blt_strategy()
        assert s.sensitivity(n_steps=50, min_sep=1, max_participations=1) > 0


class TestBltPld:
    delta = 1e-5

    def test_blt_pld(self):
        s = blt_strategy(momentum=0.95)
        eps = ftrl_acc.mf_gaussian(1.0, s, **_PART).epsilon_at(self.delta)
        assert eps > 0

    def test_blt_bnb(self):
        s = blt_strategy(momentum=0.95)
        eps = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_gaussian(1.0, s),
            num_bins=25,
            n_steps=100,
        ).epsilon_at(self.delta)
        assert eps > 0
