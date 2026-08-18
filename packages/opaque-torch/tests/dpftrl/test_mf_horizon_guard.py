"""Torch horizon guard tests for MF noise execution."""

import pytest
import torch

from opaque.api.dpftrl.noise._engine import (
    _check_mf_horizon,
    _matrix_factorization_noise,
)
from opaque.api.dpftrl.noise._lambda_cgd import _column_norm
from opaque.api.dpftrl.noise._plan import identity_execution_plan
from opaque.dpftrl.noise import (
    band_mf_strategy,
    blt_strategy,
    identity_strategy,
    lambda_cgd_strategy,
    mf_gaussian_noise,
)
from opaque.random import key
from opaque.types import clipped

_HORIZON_MATCH = r"outside the calibrated horizon"


def _zero_grads(dim: int = 32):
    return clipped({"w": torch.zeros(dim)}, max_norm=1.0)


def test_check_mf_horizon_accepts_in_range_and_rejects_past_horizon():
    for step in range(5):
        _check_mf_horizon(step, 5)
    with pytest.raises(ValueError, match=_HORIZON_MATCH):
        _check_mf_horizon(5, 5)
    with pytest.raises(ValueError, match=_HORIZON_MATCH):
        _check_mf_horizon(6, 5)


def test_column_norm_rejects_out_of_horizon_steps():
    assert all(_column_norm(0.9, 5, step) > 0.0 for step in range(5))
    with pytest.raises(ValueError, match=r"column-norm step"):
        _column_norm(0.9, 5, 5)
    with pytest.raises(ValueError, match=r"column-norm step"):
        _column_norm(0.9, 5, 6)


@pytest.mark.parametrize("normalized", [True, False])
def test_lambda_cgd_raises_on_nth_plus_one_call(normalized):
    n_steps = 4
    noise_fn, state = mf_gaussian_noise(
        {"w": torch.zeros(32)},
        lambda_cgd_strategy(lambda_=0.9, normalized=normalized),
        n_steps=n_steps,
        noise_multiplier=1.0,
        key=key(0),
    )
    for _ in range(n_steps):
        _, state = noise_fn(_zero_grads(), state)
    with pytest.raises(ValueError, match=_HORIZON_MATCH):
        noise_fn(_zero_grads(), state)


def test_normalized_lambda_cgd_never_emits_zero_noise_inside_horizon():
    n_steps = 5
    noise_fn, state = mf_gaussian_noise(
        {"w": torch.zeros(1000)},
        lambda_cgd_strategy(lambda_=0.9, normalized=True),
        n_steps=n_steps,
        noise_multiplier=1.0,
        key=key(7),
    )
    grads = _zero_grads(1000)
    for _ in range(n_steps):
        output, state = noise_fn(grads, state)
        assert output.pytree["w"].abs().max().item() > 0.0
        assert float(output.noise_stddev) > 0.0


@pytest.mark.parametrize(
    "strategy_factory",
    [
        identity_strategy,
        lambda: blt_strategy(max_buffers=3),
        lambda: band_mf_strategy(bands=3),
    ],
    ids=["identity", "blt", "band_mf"],
)
def test_public_strategies_raise_past_horizon(strategy_factory):
    n_steps = 5
    noise_fn, state = mf_gaussian_noise(
        {"w": torch.zeros(32)},
        strategy_factory(),
        n_steps=n_steps,
        noise_multiplier=1.0,
        key=key(0),
    )
    for _ in range(n_steps):
        _, state = noise_fn(_zero_grads(), state)
    with pytest.raises(ValueError, match=_HORIZON_MATCH):
        noise_fn(_zero_grads(), state)


def test_plan_engine_raises_past_plan_horizon():
    plan = identity_execution_plan(3)
    noise_fn, state = _matrix_factorization_noise(
        {"w": torch.zeros(8)}, plan, key=key(0)
    )
    for _ in range(3):
        _, state = noise_fn({"w": torch.zeros(8)}, state, stddev=1.0)
    with pytest.raises(ValueError, match=_HORIZON_MATCH):
        noise_fn({"w": torch.zeros(8)}, state, stddev=1.0)


def test_explicit_horizon_must_match_plan():
    with pytest.raises(ValueError, match=r"does not match execution-plan horizon"):
        _matrix_factorization_noise(
            {"w": torch.zeros(8)},
            identity_execution_plan(5),
            key=key(0),
            n_steps=2,
        )


@pytest.mark.parametrize("n_steps", [2.9, 5.5, True])
def test_plan_engine_rejects_non_integer_explicit_horizon(n_steps):
    with pytest.raises(TypeError, match=r"n_steps must be an int"):
        _matrix_factorization_noise(
            {"w": torch.zeros(8)},
            identity_execution_plan(5),
            key=key(0),
            n_steps=n_steps,
        )


def test_public_factory_rejects_float_horizon():
    with pytest.raises(TypeError, match=r"n_steps must be an int"):
        mf_gaussian_noise(
            {"w": torch.zeros(8)},
            lambda_cgd_strategy(lambda_=0.9),
            n_steps=4.2,
            noise_multiplier=1.0,
            key=key(0),
        )
