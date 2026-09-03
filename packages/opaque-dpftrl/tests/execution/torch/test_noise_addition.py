"""Torch tests for matrix-factorization noise execution."""

import numpy as np
import torch

from opaque.api.dpftrl.noise._engine import _matrix_factorization_noise
from opaque.api.dpftrl.noise._plan import MfExecutionPlan, toeplitz_execution_plan
from opaque.random import key


def _toeplitz_plan(inverse_coefficients):
    inverse = np.asarray(inverse_coefficients, dtype=np.float64)
    strategy = np.zeros_like(inverse)
    strategy[0] = 1.0 / inverse[0]
    for step in range(1, len(inverse)):
        strategy[step] = (
            -np.dot(inverse[1 : step + 1], strategy[step - 1 :: -1]) / inverse[0]
        )
    return toeplitz_execution_plan(strategy)


def _empirical_sequence_covariance(plan: MfExecutionPlan) -> torch.Tensor:
    grad = torch.zeros(512, dtype=torch.float64)
    sequences = []
    for seed in range(64):
        noise_fn, state = _matrix_factorization_noise(
            grad, plan, key=key(seed), compute_dtype=torch.float64
        )
        rows = []
        for _ in range(plan.n_steps):
            noised, state = noise_fn(grad, state, stddev=1.0)
            rows.append(noised)
        sequences.append(torch.stack(rows, dim=1))
    samples = torch.cat(sequences, dim=0)
    centered = samples - samples.mean(dim=0)
    return centered.T @ centered / (len(samples) - 1)


def test_toeplitz_plan_sequence_covariance_matches_recurrence():
    inverse = np.asarray([1.0, 0.5, -0.25], dtype=np.float64)
    plan = _toeplitz_plan(inverse)
    observed = _empirical_sequence_covariance(plan)
    matrix = torch.tensor(
        [[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [-0.25, 0.5, 1.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(observed, matrix @ matrix.T, atol=0.025, rtol=0)


def test_plan_continuation_reuses_same_next_column_draw():
    grad = torch.zeros(32, dtype=torch.float64)
    plan = _toeplitz_plan([1.0, 1.0, 1.0, 1.0])
    noise_fn, state = _matrix_factorization_noise(
        grad, plan, key=key(42), compute_dtype=torch.float64
    )
    for _ in range(2):
        _, state = noise_fn(grad, state, stddev=1.0)
    continued, _ = noise_fn(grad, state, stddev=1.0)
    resumed, _ = noise_fn(grad, state, stddev=1.0)
    torch.testing.assert_close(resumed, continued)
