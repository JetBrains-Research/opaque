"""Torch end-to-end tests for DP-FTRL noise in training loops."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

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


def _make_noise(model, strategy, *, steps, seed=42, min_sep=1, max_participations=1):
    template = {i: torch.zeros_like(p) for i, p in enumerate(model.parameters())}
    return mf_gaussian_noise(
        template,
        strategy,
        n_steps=steps,
        min_sep=min_sep,
        max_participations=max_participations,
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
    ("strategy_factory", "participation"),
    [
        (identity_strategy, {}),
        (lambda: band_mf_strategy(bands=10, momentum=0.0), {}),
        (lambda: lambda_cgd_strategy(lambda_=0.9), {}),
        # BiSR calibrates against a multi-epoch participation pattern:
        # 5 participations at least 10 steps apart over the 50-step run.
        (lambda: bisr_strategy(bandwidth=4), {"min_sep": 10, "max_participations": 5}),
    ],
    ids=["identity", "band_mf", "lambda_cgd", "bisr"],
)
def test_public_strategy_trains_simple_regression(strategy_factory, participation):
    x, y = _make_problem()
    model = nn.Linear(5, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    noise_fn, state = _make_noise(model, strategy_factory(), steps=50, **participation)
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
    # BnB participation: one draw per epoch, epochs num_bins steps apart —
    # the noise calibration must describe the same participation pattern
    # the sampler below actually produces.
    noise_fn, state = _make_noise(
        model,
        blt_strategy(momentum=0.9),
        steps=total_steps,
        min_sep=num_bins,
        max_participations=num_epochs,
    )
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
