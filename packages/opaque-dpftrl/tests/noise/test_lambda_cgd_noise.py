"""Provider-free tests for lambda-CGD strategy math and accounting."""

import numpy as np
import pytest

import opaque.dpftrl.accounting as ftrl_acc
from opaque.api.dpftrl.noise._lambda_cgd import LambdaCgdStrategy, lambda_cgd_strategy
from opaque.api.dpftrl.noise._toeplitz import materialize_lower_triangular
from opaque.serialization import state_dict

_PARTICIPATION = {"n_steps": 100, "min_sep": 25, "max_participations": 4}


class TestLambdaCgdStrategy:
    def test_returns_correct_type(self):
        assert isinstance(lambda_cgd_strategy(lambda_=0.9), LambdaCgdStrategy)

    def test_sensitivity_positive(self):
        assert lambda_cgd_strategy(lambda_=0.9).sensitivity(**_PARTICIPATION) > 0

    def test_gram_matrix_present(self):
        gram = lambda_cgd_strategy(lambda_=0.9).gram_matrix(**_PARTICIPATION)
        assert gram is not None
        assert len(gram) == 25 * 25

    def test_schedule_weighted_gram_matches_dense_step_weighted_operator(self):
        n_steps, min_sep, max_participations = 6, 2, 3
        learning_rates = np.asarray([1.0, 0.5, 2.0, 1.5, 0.25, 3.0])
        strategy = lambda_cgd_strategy(
            lambda_=0.4,
            normalized=False,
            lr_schedule=lambda step: float(learning_rates[step]),
        )
        encoder = materialize_lower_triangular(
            strategy.coefficients(n_steps=n_steps), n_steps
        )
        grouped_columns = np.stack(
            [
                encoder[:, bin_index::min_sep].sum(axis=1)
                for bin_index in range(min_sep)
            ],
            axis=1,
        )
        expected = grouped_columns.T @ np.diag(learning_rates) ** 2 @ grouped_columns
        gram = np.asarray(
            strategy.gram_matrix(
                n_steps=n_steps,
                min_sep=min_sep,
                max_participations=max_participations,
            )
        ).reshape(min_sep, min_sep)
        np.testing.assert_allclose(gram, expected)

    def test_uniform_schedule_matches_unweighted_gram(self):
        kwargs = {"n_steps": 12, "min_sep": 3, "max_participations": 4}
        unweighted = lambda_cgd_strategy(lambda_=0.4, normalized=False)
        weighted = lambda_cgd_strategy(
            lambda_=0.4,
            normalized=False,
            lr_schedule=lambda _step: 1.0,
        )
        assert weighted.gram_matrix(**kwargs) == pytest.approx(
            unweighted.gram_matrix(**kwargs)
        )

    def test_callable_schedule_is_not_serializable(self):
        strategy = lambda_cgd_strategy(lambda_=0.4, lr_schedule=lambda _step: 1.0)
        with pytest.raises(TypeError, match="callable strategy field"):
            state_dict(strategy)

    def test_normalized_single_participation_sensitivity_one(self):
        assert lambda_cgd_strategy(lambda_=0.9).sensitivity(
            n_steps=100, min_sep=1, max_participations=1
        ) == pytest.approx(1.0, abs=1e-6)

    def test_rejects_invalid_lambda(self):
        with pytest.raises(ValueError, match=r"lambda_ must be in \[0, 1\)"):
            lambda_cgd_strategy(lambda_=-0.1)
        with pytest.raises(ValueError, match=r"lambda_ must be in \[0, 1\)"):
            lambda_cgd_strategy(lambda_=1.0)

    def test_momentum_not_accepted(self):
        with pytest.raises(TypeError):
            lambda_cgd_strategy(lambda_=0.5, momentum=0.95)

    def test_unnormalized(self):
        assert (
            lambda_cgd_strategy(lambda_=0.9, normalized=False).sensitivity(
                **_PARTICIPATION
            )
            > 0
        )

    def test_internal_fields(self):
        strategy = lambda_cgd_strategy(lambda_=0.9)
        assert strategy.lambda_ == pytest.approx(0.9)
        assert strategy.normalized is True


class TestLambdaCgdPld:
    delta = 1e-5

    def test_lambda_cgd_pld(self):
        eps = ftrl_acc.mf_gaussian(
            1.0, lambda_cgd_strategy(lambda_=0.9), **_PARTICIPATION
        ).epsilon_at(self.delta)
        assert eps > 0

    def test_lambda_cgd_bnb(self):
        eps = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_gaussian(1.0, lambda_cgd_strategy(lambda_=0.9)),
            num_bins=25,
            n_steps=100,
        ).epsilon_at(
            1e-2,
            mc_resolution=5e-3,
            mc_failure_probability=1e-2,
        )
        assert eps > 0
