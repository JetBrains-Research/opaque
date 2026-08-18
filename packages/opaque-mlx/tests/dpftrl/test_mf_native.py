"""Native MLX parity contracts for DP-FTRL matrix-factorization noise."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from opaque import ops
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
from opaque.mlx import mlx_backend
from opaque.random import key
from opaque.serialization import from_state_dict, state_dict
from opaque.types import PerGroup, SecondMomentClippingOutput, clipped

mx = pytest.importorskip("mlx.core")


StrategyFactory = Callable[[], object]


STRATEGIES: list[pytest.ParamSpecArg] = [
    pytest.param(identity_strategy, id="identity"),
    pytest.param(lambda: band_mf_strategy(bands=3, momentum=0.5), id="band_mf"),
    pytest.param(lambda: blt_strategy(momentum=0.5), id="blt"),
    pytest.param(lambda: bisr_strategy(bandwidth=3, momentum=0.5), id="bisr"),
    pytest.param(lambda: bsr_strategy(bandwidth=3, alpha=1.0, beta=0.5), id="bsr"),
    pytest.param(lambda: lambda_cgd_strategy(lambda_=0.5), id="lambda_cgd"),
]


@pytest.fixture(autouse=True)
def _mlx_backend() -> None:
    clear_backend()
    mlx_backend()
    yield
    clear_backend()


def _make_noise(
    strategy: object,
    *,
    seed: int = 17,
    n_steps: int = 8,
    template=None,
    second_moment_strategy: object | None = None,
):
    if template is None:
        template = {"w": mx.zeros(128, dtype=mx.float32)}
    return mf_gaussian_noise(
        template,
        strategy,
        n_steps=n_steps,
        min_sep=n_steps,
        max_participations=1,
        noise_multiplier=1.25,
        key=key(seed),
        compute_dtype=mx.float32,
        second_moment_strategy=second_moment_strategy,
    )


def _is_finite(value) -> bool:
    return bool(ops.scalar_item(ops.all(ops.isfinite(value))))


def _is_equal(left, right) -> bool:
    return float(ops.scalar_item(ops.sum(ops.square(ops.subtract(left, right))))) == 0.0


@pytest.mark.parametrize("strategy_factory", STRATEGIES)
def test_native_output_replay_step_divergence_and_row_stddev(
    strategy_factory: StrategyFactory,
) -> None:
    n_steps = 8
    strategy = strategy_factory()
    template = {"w": mx.zeros(128, dtype=mx.float16)}
    grads = clipped(template, max_norm=0.4)
    noise_fn, state = _make_noise(strategy, n_steps=n_steps, template=template)

    first, state = noise_fn(grads, state)
    second, state = noise_fn(grads, state)
    mx.eval(first.pytree["w"], second.pytree["w"])

    assert isinstance(first.pytree["w"], mx.array)
    assert first.pytree["w"].dtype == template["w"].dtype
    assert first.pytree["w"].__dlpack_device__() == template["w"].__dlpack_device__()
    assert _is_finite(first.pytree["w"])
    assert _is_finite(second.pytree["w"])
    assert not _is_equal(first.pytree["w"], second.pytree["w"])
    assert state._step_counter == 2

    plan = strategy.execution_plan(
        n_steps=n_steps, min_sep=n_steps, max_participations=1
    )
    for step, output in enumerate((first, second)):
        expected = 1.25 * 0.4 * plan.row_l2[step]
        assert output.noise_stddev == pytest.approx(expected, rel=1e-6)

    replay_fn, replay_state = _make_noise(
        strategy_factory(), n_steps=n_steps, template=template
    )
    replay, replay_state = replay_fn(grads, replay_state)
    mx.eval(replay.pytree["w"])
    assert _is_equal(first.pytree["w"], replay.pytree["w"])
    assert replay_state._step_counter == 1


@pytest.mark.parametrize("strategy_factory", STRATEGIES)
def test_calibrated_horizon_rejects_nth_plus_one_call(
    strategy_factory: StrategyFactory,
) -> None:
    n_steps = 3
    template = {"w": mx.zeros(16, dtype=mx.float32)}
    grads = clipped(template, max_norm=1.0)
    noise_fn, state = _make_noise(
        strategy_factory(), n_steps=n_steps, template=template
    )

    for _ in range(n_steps):
        _, state = noise_fn(grads, state)
    with pytest.raises(ValueError, match="outside the calibrated horizon"):
        noise_fn(grads, state)


@pytest.mark.parametrize("strategy_factory", STRATEGIES)
def test_per_group_noise_runs_all_strategies(
    strategy_factory: StrategyFactory,
) -> None:
    template = {
        "small": mx.zeros(2048, dtype=mx.float32),
        "large": mx.zeros(2048, dtype=mx.float32),
    }
    bounds = PerGroup(
        groups={"small": "small", "large": "large"},
        values={"small": 0.5, "large": 2.0},
    )
    noise_fn, state = _make_noise(strategy_factory(), template=template)

    output, _ = noise_fn(clipped(template, max_norm=bounds), state)
    mx.eval(*output.pytree.values())

    assert isinstance(output.noise_stddev, PerGroup)
    assert output.noise_stddev.groups == bounds.groups
    assert output.noise_stddev.values["large"] > output.noise_stddev.values["small"]
    assert all(isinstance(value, mx.array) for value in output.pytree.values())
    assert all(_is_finite(value) for value in output.pytree.values())
    small_energy = float(ops.scalar_item(ops.mean(ops.square(output.pytree["small"]))))
    large_energy = float(ops.scalar_item(ops.mean(ops.square(output.pytree["large"]))))
    assert large_energy > small_energy


@pytest.mark.parametrize("strategy_factory", STRATEGIES)
def test_private_second_moments_are_native_finite_and_independent(
    strategy_factory: StrategyFactory,
) -> None:
    template = {"w": mx.zeros(128, dtype=mx.float32)}
    strategy = strategy_factory()
    noise_fn, state = _make_noise(
        strategy,
        template=template,
        second_moment_strategy=strategy_factory(),
    )
    paired = SecondMomentClippingOutput(
        grads=clipped(template, max_norm=0.2),
        squared_grads=clipped(template, max_norm=0.04),
    )

    output, state = noise_fn(paired, state)
    noisy_grads = output.noisy_grads.pytree["w"]
    noisy_squared_grads = output.noisy_squared_grads.pytree["w"]
    mx.eval(noisy_grads, noisy_squared_grads)

    assert isinstance(noisy_grads, mx.array)
    assert isinstance(noisy_squared_grads, mx.array)
    assert noisy_grads.dtype == template["w"].dtype
    assert noisy_squared_grads.dtype == template["w"].dtype
    assert _is_finite(noisy_grads)
    assert _is_finite(noisy_squared_grads)
    assert not _is_equal(noisy_grads, noisy_squared_grads)
    assert state._step_counter == 1


@pytest.mark.parametrize("strategy_factory", STRATEGIES)
def test_strategy_serialization_after_mlx_activation(
    strategy_factory: StrategyFactory,
) -> None:
    strategy = strategy_factory()

    restored = from_state_dict(strategy_factory(), state_dict(strategy))

    assert restored == strategy


def test_mf_state_serialization_continues_native_stream() -> None:
    template = {"w": mx.zeros(32, dtype=mx.float32)}
    grads = clipped(template, max_norm=1.0)
    strategy = band_mf_strategy(bands=3, momentum=0.8)
    noise_fn, state = _make_noise(strategy, seed=42, template=template)
    _, state = noise_fn(grads, state)
    _, state = noise_fn(grads, state)

    snapshot = state_dict(state)
    _, restore_template = _make_noise(
        band_mf_strategy(bands=3, momentum=0.8), seed=99, template=template
    )
    restored = from_state_dict(restore_template, snapshot)
    expected, expected_state = noise_fn(grads, state)
    actual, actual_state = noise_fn(grads, restored)
    mx.eval(expected.pytree["w"], actual.pytree["w"])

    assert isinstance(actual.pytree["w"], mx.array)
    assert _is_equal(actual.pytree["w"], expected.pytree["w"])
    assert actual_state._step_counter == expected_state._step_counter
