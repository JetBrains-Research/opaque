"""Torch tests for matrix-factorization noise execution."""

import numpy as np
import torch

from opaque.api.dpftrl.noise._engine import MFNoiseState, _matrix_factorization_noise
from opaque.api.dpftrl.noise._plan import MfExecutionPlan, toeplitz_execution_plan
from opaque.dpftrl.noise import band_mf_strategy, identity_strategy, mf_gaussian_noise
from opaque.random import key
from opaque.types import clipped


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


def _public_noise(template, strategy, *, n_steps=5, noise_multiplier=1.0, seed=42):
    return mf_gaussian_noise(
        template,
        strategy,
        n_steps=n_steps,
        noise_multiplier=noise_multiplier,
        key=key(seed),
    )


def test_identity_noise_preserves_shape_and_advances_state():
    grad = torch.zeros(10, dtype=torch.float64)
    noise_fn, state = _public_noise(grad, identity_strategy(), n_steps=3)
    for expected_step in range(1, 4):
        noised, state = noise_fn(clipped(grad, max_norm=1.0), state)
        assert noised.pytree.shape == grad.shape
        assert state._step_counter == expected_step


def test_identity_noise_adds_noise_and_is_stateful():
    grad = torch.zeros(10)
    noise_fn, state = _public_noise(grad, identity_strategy())
    first, state = noise_fn(clipped(grad, max_norm=1.0), state)
    second, _ = noise_fn(clipped(grad, max_norm=1.0), state)
    assert not torch.allclose(first.pytree, grad)
    assert not torch.allclose(first.pytree, second.pytree)


def test_noise_scale_is_proportional_to_multiplier():
    grad = torch.zeros(1000)
    large_fn, large_state = _public_noise(
        grad, identity_strategy(), noise_multiplier=100.0, seed=0
    )
    small_fn, small_state = _public_noise(
        grad, identity_strategy(), noise_multiplier=1.0, seed=1
    )
    large, _ = large_fn(clipped(grad, max_norm=1.0), large_state)
    small, _ = small_fn(clipped(grad, max_norm=1.0), small_state)
    assert large.pytree.std().item() > small.pytree.std().item() * 10


def test_band_mf_noise_runs_for_full_horizon():
    grad = torch.zeros(10)
    noise_fn, state = _public_noise(grad, band_mf_strategy(bands=3), n_steps=5)
    for _ in range(5):
        noised, state = noise_fn(clipped(grad, max_norm=1.0), state)
        assert noised.pytree.shape == grad.shape


def test_pytree_grads_preserve_structure_and_shapes():
    grad = {"weight": torch.zeros(5, 3), "bias": torch.zeros(3)}
    noise_fn, state = _public_noise(grad, identity_strategy())
    noised, _ = noise_fn(clipped(grad, max_norm=1.0), state)
    assert isinstance(noised.pytree, dict)
    assert noised.pytree["weight"].shape == (5, 3)
    assert noised.pytree["bias"].shape == (3,)


def test_public_factory_returns_mf_noise_state():
    _, state = _public_noise(torch.zeros(10), identity_strategy())
    assert isinstance(state, MFNoiseState)


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
