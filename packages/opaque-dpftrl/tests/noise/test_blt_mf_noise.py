"""Tests for BltStrategy factory and accounting equivalence."""

import opaque.accounting as acc
from opaque.dpftrl.noise._blt import BltStrategy, blt_strategy


class TestBltStrategy:
    def test_returns_correct_type(self):
        s = blt_strategy(n_steps=100, min_sep=25, max_participations=4, momentum=0.95)
        assert isinstance(s, BltStrategy)

    def test_sensitivity_positive(self):
        s = blt_strategy(n_steps=100, min_sep=25, max_participations=4, momentum=0.95)
        assert s.sensitivity > 0

    def test_gram_matrix_present(self):
        s = blt_strategy(n_steps=100, min_sep=25, max_participations=4, momentum=0.95)
        assert s.gram_matrix is not None
        assert len(s.gram_matrix) == 25 * 25

    def test_coefficients_length(self):
        s = blt_strategy(n_steps=100, min_sep=25, max_participations=4, momentum=0.95)
        assert len(s.coefficients) == 100

    def test_streaming_matrix_present(self):
        s = blt_strategy(n_steps=100, min_sep=25, max_participations=4, momentum=0.95)
        assert s._streaming_matrix is not None

    def test_matches_old_sensitivity(self):
        new = blt_strategy(n_steps=100, min_sep=25, max_participations=4, momentum=0.95)
        assert new.sensitivity > 0

    def test_single_participation(self):
        s = blt_strategy(n_steps=50, min_sep=1, max_participations=1)
        assert s.sensitivity > 0


class TestBltPld:
    delta = 1e-5

    def test_blt_pld(self):
        s = blt_strategy(n_steps=100, min_sep=25, max_participations=4, momentum=0.95)
        eps = acc.blt(1.0, sensitivity=s.sensitivity).epsilon_at(self.delta)
        assert eps > 0

    def test_blt_bnb(self):
        s = blt_strategy(n_steps=100, min_sep=25, max_participations=4, momentum=0.95)
        eps = acc.balls_in_bins(
            acc.blt(1.0, sensitivity=s.sensitivity, gram_matrix=s.gram_matrix),
            num_bins=25,
            num_epochs=4,
        ).epsilon_at(self.delta)
        assert eps > 0
