"""Tests for LrAwareStrategy and schedule workload error (arXiv:2511.17994)."""

import math

import pytest
import torch

import opaque_accounting as acc
from opaque.noise.mf.lr_aware import LrAwareStrategy, lr_aware_strategy
from opaque.noise.mf._toeplitz import (
    schedule_per_query_error,
    schedule_max_error,
    schedule_mean_error,
    per_query_error,
    sensitivity_squared,
)
from opaque.noise.mf.bsr import _r_sequence


class TestLrAwareCoefficients:
    def test_c_alpha_matches_r_sequence_scaled(self):
        """C_alpha[j] = alpha^j * r_j (paper eq. 8)."""
        n = 100
        beta = 0.25
        alpha = beta ** (1.0 / (n - 1))
        s = lr_aware_strategy(bandwidth=8, n_steps=n, min_sep=20, lr_decay_beta=beta)
        r = _r_sequence(8)
        for j in range(8):
            expected = alpha**j * r[j]
            assert s.coefficients[j] == pytest.approx(expected, rel=1e-10)

    def test_coefficients_are_non_negative(self):
        s = lr_aware_strategy(bandwidth=16, n_steps=200, min_sep=50, lr_decay_beta=0.1)
        for c in s.coefficients[:16]:
            assert c >= 0.0

    def test_coefficients_are_non_increasing(self):
        s = lr_aware_strategy(bandwidth=16, n_steps=200, min_sep=50, lr_decay_beta=0.25)
        for i in range(1, 16):
            assert s.coefficients[i] <= s.coefficients[i - 1] + 1e-12


class TestLrAwareStrategy:
    def test_returns_correct_type(self):
        s = lr_aware_strategy(
            bandwidth=4, n_steps=100, min_sep=25,
            max_participations=4, lr_decay_beta=0.5,
        )
        assert isinstance(s, LrAwareStrategy)

    def test_sensitivity_positive(self):
        s = lr_aware_strategy(
            bandwidth=4, n_steps=100, min_sep=25,
            max_participations=4, lr_decay_beta=0.5,
        )
        assert s.sensitivity > 0

    def test_gram_matrix_present(self):
        s = lr_aware_strategy(
            bandwidth=4, n_steps=100, min_sep=25,
            max_participations=4, lr_decay_beta=0.5,
        )
        assert s.gram_matrix is not None
        assert len(s.gram_matrix) == 25 * 25

    def test_streaming_matrix_present(self):
        s = lr_aware_strategy(
            bandwidth=4, n_steps=100, min_sep=25,
            max_participations=4, lr_decay_beta=0.5,
        )
        assert s._streaming_matrix is not None

    def test_rejects_beta_out_of_range(self):
        with pytest.raises(ValueError):
            lr_aware_strategy(bandwidth=4, n_steps=100, min_sep=25, lr_decay_beta=0.0)
        with pytest.raises(ValueError):
            lr_aware_strategy(bandwidth=4, n_steps=100, min_sep=25, lr_decay_beta=1.0)
        with pytest.raises(ValueError):
            lr_aware_strategy(bandwidth=4, n_steps=100, min_sep=25, lr_decay_beta=-0.1)


class TestScheduleWorkloadError:
    def test_constant_schedule_matches_prefix_sum(self):
        """A_chi with chi=1 everywhere is the prefix-sum workload."""
        n = 20
        coef = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float64)
        lr_const = torch.ones(n, dtype=torch.float64)
        sq = schedule_per_query_error(strategy_coef=coef, n=n, lr_schedule=lr_const)
        toep = per_query_error(strategy_coef=coef, n=n)
        torch.testing.assert_close(sq, toep, atol=1e-10, rtol=1e-10)

    def test_exponential_schedule_lower_error(self):
        """Decaying LR should produce lower total error than constant LR."""
        n = 32
        coef = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float64)
        lr_const = torch.ones(n, dtype=torch.float64)
        lr_exp = torch.tensor(
            [0.5 ** (t / (n - 1)) for t in range(n)], dtype=torch.float64
        )
        err_const = schedule_mean_error(strategy_coef=coef, n=n, lr_schedule=lr_const)
        err_exp = schedule_mean_error(strategy_coef=coef, n=n, lr_schedule=lr_exp)
        assert err_exp < err_const

    def test_lr_aware_vs_prefix_sum_on_schedule_workload(self):
        """LR-aware C_alpha should have lower schedule-workload error than
        prefix-sum C (optimal_max_error_strategy_coefs) for exponential decay."""
        from opaque.noise.mf._toeplitz import optimal_max_error_strategy_coefs

        n = 64
        beta = 0.25
        lr_schedule = torch.tensor(
            [beta ** (t / (n - 1)) for t in range(n)], dtype=torch.float64
        )

        prefix_coef = optimal_max_error_strategy_coefs(8)
        s_lr = lr_aware_strategy(bandwidth=8, n_steps=n, min_sep=n, lr_decay_beta=beta)
        lr_coef = torch.tensor(s_lr.coefficients[:8], dtype=torch.float64)

        err_prefix = float(
            schedule_mean_error(strategy_coef=prefix_coef, n=n, lr_schedule=lr_schedule)
            * sensitivity_squared(prefix_coef, n)
        )
        err_lr = float(
            schedule_mean_error(strategy_coef=lr_coef, n=n, lr_schedule=lr_schedule)
            * sensitivity_squared(lr_coef, n)
        )
        assert err_lr <= err_prefix * 1.05


class TestLrAwarePld:
    delta = 1e-5

    def test_lr_aware_pld(self):
        s = lr_aware_strategy(
            bandwidth=4, n_steps=100, min_sep=25,
            max_participations=4, lr_decay_beta=0.5,
        )
        eps = acc.lr_aware(1.0, sensitivity=s.sensitivity).epsilon_at(self.delta)
        assert eps > 0

    def test_lr_aware_bnb(self):
        s = lr_aware_strategy(
            bandwidth=4, n_steps=100, min_sep=25,
            max_participations=4, lr_decay_beta=0.5,
        )
        eps = acc.balls_in_bins(
            acc.lr_aware(
                1.0, sensitivity=s.sensitivity, gram_matrix=s.gram_matrix
            ),
            num_bins=25,
            num_epochs=4,
        ).epsilon_at(self.delta)
        assert eps > 0
