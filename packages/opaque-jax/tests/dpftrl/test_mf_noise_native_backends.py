"""Native JAX execution contracts for DP-FTRL matrix-factorization noise."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opaque.api.engine.backend import clear_backend
from opaque.distributed import sync
from opaque.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    bsr_strategy,
    identity_strategy,
    lambda_cgd_strategy,
    mf_gaussian_noise,
)
from opaque.dpftrl.noise.types import MfStrategy
from opaque.random import key
from opaque.types import PerGroup, SecondMomentClippingOutput, clipped

StrategyFactory = Callable[[], MfStrategy]

_STRATEGIES = [
    pytest.param(identity_strategy, id="identity"),
    pytest.param(lambda: band_mf_strategy(bands=2, momentum=0.8), id="band_mf"),
    pytest.param(lambda: blt_strategy(max_buffers=2, momentum=0.8), id="blt"),
    pytest.param(lambda: bisr_strategy(bandwidth=2, momentum=0.3), id="bisr"),
    pytest.param(lambda: bsr_strategy(bandwidth=2, alpha=1.0, beta=0.3), id="bsr"),
    pytest.param(lambda: lambda_cgd_strategy(lambda_=0.7), id="lambda_cgd"),
]


@pytest.fixture(autouse=True)
def _unselected_backend() -> None:
    clear_backend()
    yield
    clear_backend()


def _make_noise(
    strategy_factory: StrategyFactory,
    template: Any,
    *,
    n_steps: int = 4,
    seed: int = 17,
    second_moment_strategy: MfStrategy | None = None,
):
    return mf_gaussian_noise(
        template,
        strategy_factory(),
        n_steps=n_steps,
        min_sep=1,
        max_participations=1,
        noise_multiplier=1.25,
        key=key(seed),
        compute_dtype=jnp.float32,
        second_moment_strategy=second_moment_strategy,
    )


@pytest.mark.parametrize("strategy_factory", _STRATEGIES)
def test_all_strategies_preserve_native_output_and_replay(
    strategy_factory: StrategyFactory,
) -> None:
    device = jax.devices()[0]
    template = {
        "weight": jax.device_put(jnp.zeros((8, 8), dtype=jnp.float16), device),
        "bias": jax.device_put(jnp.zeros(8, dtype=jnp.float32), device),
    }
    grads = clipped(template, max_norm=0.75)

    noise_fn, state = _make_noise(strategy_factory, template)
    first, state = noise_fn(grads, state)
    second, state = noise_fn(grads, state)

    replay_fn, replay_state = _make_noise(strategy_factory, template)
    replay_first, replay_state = replay_fn(grads, replay_state)
    replay_second, replay_state = replay_fn(grads, replay_state)

    for name, value in first.pytree.items():
        assert isinstance(value, jax.Array)
        assert value.dtype == template[name].dtype
        assert value.device == template[name].device
        assert bool(jnp.all(jnp.isfinite(value)))
        np.testing.assert_array_equal(value, replay_first.pytree[name])
        np.testing.assert_array_equal(second.pytree[name], replay_second.pytree[name])

    assert not np.array_equal(first.pytree["weight"], second.pytree["weight"])
    assert state._step_counter == replay_state._step_counter == 2
    assert sync(state) is state


@pytest.mark.parametrize("strategy_factory", _STRATEGIES)
def test_all_strategies_publish_realized_row_stddev_and_guard_horizon(
    strategy_factory: StrategyFactory,
) -> None:
    n_steps = 3
    max_norm = 0.4
    noise_multiplier = 1.25
    strategy = strategy_factory()
    plan = strategy.execution_plan(n_steps=n_steps, min_sep=1, max_participations=1)
    template = jnp.zeros(32, dtype=jnp.float32)
    noise_fn, state = mf_gaussian_noise(
        template,
        strategy,
        n_steps=n_steps,
        min_sep=1,
        max_participations=1,
        noise_multiplier=noise_multiplier,
        key=key(23),
    )
    grads = clipped(template, max_norm=max_norm)

    for step in range(n_steps):
        output, state = noise_fn(grads, state)
        assert output.noise_stddev == pytest.approx(
            noise_multiplier * max_norm * plan.row_l2[step]
        )
        assert bool(jnp.all(jnp.isfinite(output.pytree)))

    with pytest.raises(ValueError, match="outside the calibrated horizon"):
        noise_fn(grads, state)


def test_per_group_noise_uses_native_jax_arrays_and_realized_stddevs() -> None:
    template = {
        "small": jnp.zeros(4096, dtype=jnp.float32),
        "large": jnp.zeros(4096, dtype=jnp.float32),
    }
    bounds = PerGroup(
        groups={"small": "small", "large": "large"},
        values={"small": 1.0, "large": 4.0},
    )
    noise_fn, state = _make_noise(identity_strategy, template)

    output, _ = noise_fn(clipped(template, max_norm=bounds), state)

    assert isinstance(output.noise_stddev, PerGroup)
    assert output.noise_stddev.values == {
        "small": pytest.approx(1.25 * np.sqrt(5.0)),
        "large": pytest.approx(1.25 * np.sqrt(20.0)),
    }
    assert all(isinstance(value, jax.Array) for value in output.pytree.values())
    small_variance = float(jnp.mean(jnp.square(output.pytree["small"])))
    large_variance = float(jnp.mean(jnp.square(output.pytree["large"])))
    assert large_variance > 2.5 * small_variance


def test_paired_private_second_moments_are_native_finite_and_independent() -> None:
    template = {
        "weight": jnp.zeros((8, 8), dtype=jnp.float32),
        "bias": jnp.zeros(8, dtype=jnp.float32),
    }
    first_bounds = PerGroup(
        groups={"weight": "weight", "bias": "bias"},
        values={"weight": 0.1, "bias": 0.2},
    )
    second_bounds = first_bounds * first_bounds
    paired = SecondMomentClippingOutput(
        clipped(template, max_norm=first_bounds),
        clipped(template, max_norm=second_bounds),
    )
    second_strategy = band_mf_strategy(bands=2, momentum=0.9)
    noise_fn, state = _make_noise(
        lambda: band_mf_strategy(bands=2, momentum=0.8),
        template,
        second_moment_strategy=second_strategy,
    )

    output, state = noise_fn(paired, state)

    assert state._step_counter == 1
    assert isinstance(output.noisy_grads.noise_stddev, PerGroup)
    assert isinstance(output.noisy_squared_grads.noise_stddev, PerGroup)
    for name in template:
        noisy_grad = output.noisy_grads.pytree[name]
        noisy_square = output.noisy_squared_grads.pytree[name]
        assert isinstance(noisy_grad, jax.Array)
        assert isinstance(noisy_square, jax.Array)
        assert noisy_grad.dtype == noisy_square.dtype == template[name].dtype
        assert bool(jnp.all(jnp.isfinite(noisy_grad)))
        assert bool(jnp.all(jnp.isfinite(noisy_square)))
    assert not np.array_equal(
        output.noisy_grads.pytree["weight"],
        output.noisy_squared_grads.pytree["weight"],
    )
