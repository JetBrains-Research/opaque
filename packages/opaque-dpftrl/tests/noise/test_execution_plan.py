"""Provider-independent MF execution-plan contracts."""

import numpy as np
import pytest

from opaque.api.dpftrl.noise._plan import (
    identity_execution_plan,
    lambda_replay_execution_plan,
    toeplitz_execution_plan,
)
from opaque.dpftrl.noise import (
    bisr_strategy,
    blt_strategy,
    bsr_strategy,
    identity_strategy,
    lambda_cgd_strategy,
)


def test_toeplitz_plan_matches_dense_inverse_and_row_norms():
    coefficients = np.asarray([1.0, 0.4, 0.2, 0.0], dtype=np.float64)
    plan = toeplitz_execution_plan(coefficients)
    strategy = np.zeros((4, 4), dtype=np.float64)
    for row in range(4):
        strategy[row, : row + 1] = coefficients[row::-1]
    inverse = np.linalg.inv(strategy)

    np.testing.assert_allclose(plan.inverse_coefficients, inverse[:, 0])
    np.testing.assert_allclose(plan.row_l2, np.linalg.norm(inverse, axis=1))
    assert plan.mode == "toeplitz"


def test_normalized_lambda_plan_matches_closed_form_row_norms():
    plan = lambda_replay_execution_plan(0.6, 5, normalized=True)
    expected_scales = np.asarray(
        [np.sqrt(sum(0.6 ** (2 * i) for i in range(5 - step))) for step in range(5)]
    )
    expected_rows = expected_scales * np.asarray([1.0, *([np.sqrt(1.0 + 0.6**2)] * 4)])

    np.testing.assert_allclose(plan.column_scales, expected_scales)
    np.testing.assert_allclose(plan.row_l2, expected_rows)
    assert plan.mode == "lambda_replay"


@pytest.mark.parametrize(
    "strategy",
    [
        identity_strategy(),
        bsr_strategy(bandwidth=3, alpha=1.0, beta=0.1),
        bisr_strategy(bandwidth=3),
        lambda_cgd_strategy(lambda_=0.6),
    ],
)
def test_strategy_coefficients_are_float64_numpy_arrays(strategy):
    coefficients = strategy.coefficients(n_steps=8)
    plan = strategy.execution_plan(n_steps=8)

    assert isinstance(coefficients, np.ndarray)
    assert coefficients.dtype == np.float64
    np.testing.assert_allclose(coefficients, plan.coefficients())


def test_plan_vectors_are_immutable_tuples_and_array_queries_are_copies():
    plan = identity_execution_plan(3)
    coefficients = plan.coefficients()
    coefficients[0] = 7.0

    assert plan.strategy_coefficients == (1.0, 0.0, 0.0)
    np.testing.assert_array_equal(plan.coefficients(), [1.0, 0.0, 0.0])


def test_blt_plan_preserves_buffer_parameters():
    strategy = blt_strategy(max_buffers=1)
    context = {"n_steps": 8, "min_sep": 8, "max_participations": 1}
    plan = strategy.execution_plan(**context)

    assert plan.mode == "blt"
    assert len(plan.buffer_decay) == len(plan.output_scale)
    np.testing.assert_allclose(plan.coefficients(), strategy.coefficients(**context))


@pytest.mark.parametrize(
    "factory",
    [
        identity_execution_plan,
        lambda n: lambda_replay_execution_plan(0.5, n, normalized=True),
    ],
)
def test_plan_factories_reject_empty_horizons(factory):
    with pytest.raises(ValueError, match="n_steps must be >= 1"):
        factory(0)
