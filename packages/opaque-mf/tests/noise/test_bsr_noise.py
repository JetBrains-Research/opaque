"""Tests for BsrStrategy factory and accounting."""

import pytest

import opaque_accounting as acc
from opaque.mf.noise.bsr import BsrStrategy, bsr_strategy


class TestBsrStrategy:
    def test_returns_correct_type(self):
        s = bsr_strategy(
            bandwidth=4,
            n_steps=100,
            min_sep=25,
            max_participations=4,
            alpha=1.0,
            beta=0.5,
        )
        assert isinstance(s, BsrStrategy)

    def test_sensitivity_positive(self):
        s = bsr_strategy(
            bandwidth=4,
            n_steps=100,
            min_sep=25,
            max_participations=4,
            alpha=1.0,
            beta=0.5,
        )
        assert s.sensitivity > 0

    def test_gram_matrix_present(self):
        s = bsr_strategy(
            bandwidth=4,
            n_steps=100,
            min_sep=25,
            max_participations=4,
            alpha=1.0,
            beta=0.5,
        )
        assert s.gram_matrix is not None
        assert len(s.gram_matrix) == 25 * 25

    def test_streaming_matrix_present(self):
        s = bsr_strategy(
            bandwidth=4,
            n_steps=100,
            min_sep=25,
            max_participations=4,
            alpha=1.0,
            beta=0.5,
        )
        assert s._streaming_matrix is not None

    def test_coefficients_non_increasing_first_band(self):
        s = bsr_strategy(
            bandwidth=6,
            n_steps=50,
            min_sep=10,
            max_participations=3,
            alpha=1.0,
            beta=0.9,
        )
        head = list(s.coefficients[:6])
        for i in range(1, len(head)):
            assert head[i] <= head[i - 1] + 1e-9

    def test_rejects_alpha_le_beta(self):
        with pytest.raises(ValueError, match="α > β"):
            bsr_strategy(
                bandwidth=4,
                n_steps=50,
                min_sep=10,
                max_participations=2,
                alpha=0.9,
                beta=0.95,
            )

    def test_rejects_momentum_one(self):
        with pytest.raises(ValueError):
            bsr_strategy(
                bandwidth=4,
                n_steps=50,
                min_sep=10,
                max_participations=2,
                alpha=1.0,
                beta=1.0,
            )

    def test_rejects_normalized_flag(self):
        with pytest.raises(ValueError, match="normalized=True"):
            bsr_strategy(
                bandwidth=4,
                n_steps=50,
                min_sep=10,
                max_participations=2,
                alpha=1.0,
                beta=0.5,
                normalized=True,
            )


class TestBsrPld:
    delta = 1e-5

    def test_bsr_pld(self):
        s = bsr_strategy(
            bandwidth=4,
            n_steps=100,
            min_sep=25,
            max_participations=4,
            alpha=1.0,
            beta=0.5,
        )
        eps = acc.bsr(1.0, sensitivity=s.sensitivity).epsilon_at(self.delta)
        assert eps > 0

    def test_bsr_bnb(self):
        s = bsr_strategy(
            bandwidth=4,
            n_steps=100,
            min_sep=25,
            max_participations=4,
            alpha=1.0,
            beta=0.5,
        )
        eps = acc.balls_in_bins(
            acc.bsr(1.0, sensitivity=s.sensitivity, gram_matrix=s.gram_matrix),
            num_bins=25,
            num_epochs=4,
        ).epsilon_at(self.delta)
        assert eps > 0
