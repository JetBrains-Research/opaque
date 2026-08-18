"""Torch end-to-end tests for DP-FTRL noise in training loops."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from opaque.api.dpftrl.noise._engine import _matrix_factorization_noise
from opaque.api.dpftrl.noise._plan import identity_execution_plan
from opaque.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    identity_strategy,
    lambda_cgd_strategy,
    mf_gaussian_noise,
)
from opaque.dpftrl.sampling import BallsInBinsSampler
from opaque.random import key
from opaque.types import clipped


def _make_problem(dim=5, n_samples=50, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(n_samples, dim)
    y = x @ torch.randn(dim, 1)
    return x, y


def _make_noise(model, strategy, *, steps, seed=42):
    template = {i: torch.zeros_like(p) for i, p in enumerate(model.parameters())}
    return mf_gaussian_noise(
        template,
        strategy,
        n_steps=steps,
        min_sep=1,
        max_participations=1,
        noise_multiplier=1.0,
        key=key(seed),
    )


def _train(model, optimizer, noise_fn, state, x, y, *, steps, max_norm=0.1):
    params = list(model.parameters())
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        grads = {i: p.grad.clone() for i, p in enumerate(params)}
        noised, state = noise_fn(clipped(grads, max_norm=max_norm), state)
        for index, parameter in enumerate(params):
            parameter.grad = noised.pytree[index].to(parameter.dtype)
        optimizer.step()
        losses.append(loss.item())
    return losses


@pytest.mark.parametrize(
    "strategy_factory",
    [
        identity_strategy,
        lambda: band_mf_strategy(bands=10, momentum=0.0),
        lambda: lambda_cgd_strategy(lambda_=0.9),
        lambda: bisr_strategy(bandwidth=4),
    ],
    ids=["identity", "band_mf", "lambda_cgd", "bisr"],
)
def test_public_strategy_trains_simple_regression(strategy_factory):
    x, y = _make_problem()
    model = nn.Linear(5, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    noise_fn, state = _make_noise(model, strategy_factory(), steps=50)
    losses = _train(model, optimizer, noise_fn, state, x, y, steps=50)
    assert losses[-1] < losses[0]


def test_identity_noise_trains_with_adam():
    x, y = _make_problem()
    model = nn.Linear(5, 1, bias=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    noise_fn, state = _make_noise(model, identity_strategy(), steps=30)
    losses = _train(model, optimizer, noise_fn, state, x, y, steps=30)
    assert losses[-1] < losses[0]


def test_identity_noise_trains_multi_parameter_model():
    x, y = _make_problem(dim=10)
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    noise_fn, state = _make_noise(model, identity_strategy(), steps=30)
    losses = _train(model, optimizer, noise_fn, state, x, y, steps=30)
    assert losses[-1] < losses[0]


def test_training_is_deterministic_with_same_keys():
    results = []
    for _ in range(2):
        x, y = _make_problem(n_samples=4)
        torch.manual_seed(0)
        model = nn.Linear(5, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        noise_fn, state = _make_noise(model, identity_strategy(), steps=3)
        _train(model, optimizer, noise_fn, state, x, y, steps=3, max_norm=1.0)
        results.append(model.weight.detach().clone())
    torch.testing.assert_close(results[0], results[1])


def test_band_mf_and_identity_both_train_at_same_noise_level():
    x, y = _make_problem(n_samples=100)
    losses = []
    for strategy in (identity_strategy(), band_mf_strategy(bands=10)):
        torch.manual_seed(0)
        model = nn.Linear(5, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        noise_fn, state = _make_noise(model, strategy, steps=100)
        losses.append(
            _train(model, optimizer, noise_fn, state, x, y, steps=100, max_norm=0.5)
        )
    identity_losses, band_losses = losses
    assert identity_losses[-1] < identity_losses[0]
    assert band_losses[-1] < band_losses[0]
    assert band_losses[-1] < identity_losses[0]


def test_blt_with_balls_in_bins_trains():
    x, y = _make_problem(n_samples=200)
    dataset = TensorDataset(x, y)
    num_bins, num_epochs = 10, 3
    total_steps = num_bins * num_epochs
    model = nn.Linear(5, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    noise_fn, state = _make_noise(model, blt_strategy(momentum=0.9), steps=total_steps)
    sampler = BallsInBinsSampler(
        dataset, num_bins=num_bins, n_steps=total_steps, key=key(99)
    )
    params = list(model.parameters())
    losses = []
    for indices in sampler:
        optimizer.zero_grad()
        loss = ((model(x[indices]) - y[indices]) ** 2).mean()
        loss.backward()
        grads = {i: p.grad.clone() for i, p in enumerate(params)}
        noised, state = noise_fn(clipped(grads, max_norm=0.05), state)
        for index, parameter in enumerate(params):
            parameter.grad = noised.pytree[index].to(parameter.dtype)
        optimizer.step()
        losses.append(loss.item())
    assert len(losses) == total_steps
    assert losses[-1] < losses[0]


def test_balls_in_bins_covers_dataset_and_repeats_assignment():
    n_samples, num_bins, num_epochs = 1000, 10, 3
    sampler = BallsInBinsSampler(
        list(range(n_samples)),
        num_bins=num_bins,
        n_steps=num_bins * num_epochs,
        key=key(42),
    )
    batches = list(sampler)
    first_epoch = batches[:num_bins]
    assert {index for batch in first_epoch for index in batch} == set(range(n_samples))
    assert sum(map(len, first_epoch)) == n_samples
    for epoch in range(1, num_epochs):
        assert batches[epoch * num_bins : (epoch + 1) * num_bins] == first_epoch


def test_balls_in_bins_has_variable_bin_sizes():
    sampler = BallsInBinsSampler(
        list(range(10000)), num_bins=50, n_steps=50, key=key(123)
    )
    assert len({len(batch) for batch in sampler}) > 1


def test_public_identity_matches_identity_execution_plan():
    template = {"w": torch.zeros(10)}
    public_fn, public_state = mf_gaussian_noise(
        template,
        identity_strategy(),
        n_steps=10,
        noise_multiplier=1.0,
        key=key(42),
    )
    plan_fn, plan_state = _matrix_factorization_noise(
        template, identity_execution_plan(10), key=key(42)
    )
    grad = {"w": torch.ones(10)}
    public, _ = public_fn(clipped(grad, max_norm=1.0), public_state)
    planned, _ = plan_fn(grad, plan_state, stddev=1.0)
    torch.testing.assert_close(public.pytree["w"], planned["w"])
