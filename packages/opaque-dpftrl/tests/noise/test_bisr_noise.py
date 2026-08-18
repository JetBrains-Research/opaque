"""Provider-free tests for BISR strategy math and accounting."""

import numpy as np
import pytest

import opaque.dpftrl.accounting as ftrl_acc
from opaque.api.dpftrl.noise._bisr import BisrStrategy, _native, bisr_strategy
from opaque.api.dpftrl.noise._toeplitz import (
    inverse_as_streaming_matrix,
    materialize_lower_triangular,
)
from opaque.serialization import state_dict

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

    def test_schedule_weighted_gram_matches_dense_step_weighted_operator(self):
        n_steps, min_sep, max_participations = 6, 2, 3
        learning_rates = np.asarray([1.0, 0.5, 2.0, 1.5, 0.25, 3.0])
        strategy = bisr_strategy(
            bandwidth=3,
            normalized=False,
            momentum=0.3,
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
        unweighted = bisr_strategy(bandwidth=3, normalized=False, momentum=0.3)
        weighted = bisr_strategy(
            bandwidth=3,
            normalized=False,
            momentum=0.3,
            lr_schedule=lambda _step: 1.0,
        )
        assert weighted.gram_matrix(**kwargs) == pytest.approx(
            unweighted.gram_matrix(**kwargs)
        )

    def test_callable_schedule_is_not_serializable(self):
        strategy = bisr_strategy(bandwidth=3, lr_schedule=lambda _step: 1.0)
        with pytest.raises(TypeError, match="callable strategy field"):
            state_dict(strategy)

    def test_streaming_matrix_present(self):
        assert bisr_strategy(bandwidth=4).streaming_matrix(**_PART) is not None

    @pytest.mark.parametrize("bandwidth", [2, 4])
    @pytest.mark.parametrize("n_steps", [6, 12])
    def test_execution_plan_uses_full_horizon_strategy(self, bandwidth, n_steps):
        strategy = bisr_strategy(bandwidth=bandwidth, normalized=False, momentum=0.3)
        streaming = strategy.streaming_matrix(n_steps=n_steps)
        plan = strategy.execution_plan(n_steps=n_steps)
        expected_dense = streaming.materialize(n_steps)
        np.testing.assert_allclose(
            plan.row_l2,
            np.linalg.norm(expected_dense, axis=1),
        )
        full_horizon_coefficients = _native().bisr_strategy_coefficients(
            list(strategy._inv_coefs()), n_steps
        )
        manual_streaming = inverse_as_streaming_matrix(full_horizon_coefficients)
        np.testing.assert_allclose(
            expected_dense, manual_streaming.materialize(n_steps)
        )

    def test_with_momentum(self):
        assert bisr_strategy(bandwidth=4, momentum=0.95).sensitivity(**_PART) > 0

    def test_rejects_bad_bandwidth(self):
        with pytest.raises(ValueError, match="bandwidth must be >= 2"):
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
