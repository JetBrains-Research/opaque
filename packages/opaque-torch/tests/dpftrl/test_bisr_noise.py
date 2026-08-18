"""Torch runtime tests for BISR matrix-factorization noise."""

import pytest
import torch

from opaque.dpftrl.noise import bisr_strategy, identity_strategy, mf_gaussian_noise
from opaque.random import key
from opaque.types import NoisedPytree, clipped


def _run(strategy, *, n_steps: int, seed: int = 0):
    template = {"w": torch.zeros(16, dtype=torch.float64)}
    noise_fn, state = mf_gaussian_noise(
        template,
        strategy,
        n_steps=n_steps,
        min_sep=1,
        max_participations=1,
        noise_multiplier=1.0,
        key=key(seed),
        compute_dtype=torch.float64,
    )
    grads = clipped(template, max_norm=1.0)
    rows = []
    for _ in range(n_steps):
        output, state = noise_fn(grads, state)
        assert isinstance(output, NoisedPytree)
        rows.append(output.pytree["w"])
    return torch.stack(rows), state


def test_mf_gaussian_noise_executes_bisr_plan():
    output, state = _run(bisr_strategy(bandwidth=4), n_steps=12, seed=1)

    assert output.shape == (12, 16)
    assert state._step_counter == 12
    assert output.abs().max().item() > 0.0


@pytest.mark.parametrize("bandwidth", [2, 4])
@pytest.mark.parametrize("n_steps", [6, 12])
def test_runtime_recurrence_matches_full_horizon_plan(bandwidth, n_steps):
    strategy = bisr_strategy(bandwidth=bandwidth, normalized=False, momentum=0.3)
    actual, _ = _run(strategy, n_steps=n_steps)
    iid, _ = _run(identity_strategy(), n_steps=n_steps)
    plan = strategy.execution_plan(
        n_steps=n_steps,
        min_sep=1,
        max_participations=1,
    )

    expected_rows = []
    for step in range(n_steps):
        count = min(step + 1, len(plan.inverse_coefficients))
        expected = torch.zeros_like(iid[step])
        for offset in range(count):
            expected = expected + plan.inverse_coefficients[offset] * iid[step - offset]
        expected_rows.append(expected * plan.column_scales[step])

    torch.testing.assert_close(actual, torch.stack(expected_rows))
