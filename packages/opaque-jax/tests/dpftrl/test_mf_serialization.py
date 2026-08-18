"""JAX serialization contracts for DP-FTRL strategies and MF state."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opaque.api.engine.backend import clear_backend
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
from opaque.jax import jax_backend
from opaque.random import key
from opaque.serialization import from_state_dict, state_dict
from opaque.types import clipped

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


@pytest.mark.parametrize("strategy_factory", _STRATEGIES)
def test_strategy_round_trip_after_jax_activation(
    strategy_factory: StrategyFactory,
) -> None:
    jax_backend()
    strategy = strategy_factory()

    serialized = state_dict(strategy)
    restored = from_state_dict(strategy_factory(), serialized)

    assert type(restored) is type(strategy)
    assert restored == strategy
    assert state_dict(restored) == serialized


def _make_band_mf_noise(seed: int):
    template = {"weight": jnp.zeros(16, dtype=jnp.float32)}
    noise_fn, state = mf_gaussian_noise(
        template,
        band_mf_strategy(bands=2, momentum=0.8),
        n_steps=5,
        min_sep=1,
        max_participations=1,
        noise_multiplier=1.0,
        key=key(seed),
    )
    return noise_fn, state, template


def test_mf_state_round_trip_continues_native_jax_stream() -> None:
    jax_backend()
    noise_fn, state, template = _make_band_mf_noise(seed=42)
    grads = clipped(template, max_norm=1.0)
    _, state = noise_fn(grads, state)
    _, state = noise_fn(grads, state)

    snapshot = state_dict(state)
    _, restore_template, _ = _make_band_mf_noise(seed=99)
    restored = from_state_dict(restore_template, snapshot)

    expected, expected_state = noise_fn(grads, state)
    actual, actual_state = noise_fn(grads, restored)

    assert isinstance(actual.pytree["weight"], jax.Array)
    np.testing.assert_array_equal(actual.pytree["weight"], expected.pytree["weight"])
    assert actual.pytree["weight"].device == template["weight"].device
    assert actual_state._step_counter == expected_state._step_counter == 3
