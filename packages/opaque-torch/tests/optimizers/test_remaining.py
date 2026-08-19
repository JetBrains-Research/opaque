"""Smoke coverage for all backend-neutral optimizer families."""

from __future__ import annotations

import pytest
import torch

from opaque.api.optimizers import (
    adadelta,
    adafactor,
    adagrad,
    ademamix,
    lion,
    radam,
    rmsprop,
    schedule_free,
)
from opaque.serialization import from_state_dict, state_dict


@pytest.fixture
def params() -> dict[str, torch.Tensor]:
    return {"weight": torch.ones((3, 2)), "bias": torch.ones(2)}


@pytest.fixture
def grads(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: torch.full_like(value, 0.25) for name, value in params.items()}


@pytest.mark.parametrize(
    "factory",
    [adadelta, adafactor, adagrad, ademamix, lion, radam, rmsprop],
)
def test_factory_returns_signed_updates_and_serializable_state(factory, params, grads):
    step, state = factory(params, lr=1e-2)
    updates, state = step(grads, state, params=params)

    assert state.step == 1
    for name, update in updates.items():
        assert update.shape == params[name].shape
        assert torch.isfinite(update).all()

    restored = from_state_dict(state, state_dict(state))
    assert type(restored) is type(state)
    assert restored.step == state.step


def test_schedule_free_composes_an_engine_factory(params, grads):
    step, state = schedule_free(params, adagrad, lr=1e-2)
    updates, state = step(grads, state, params=params)

    assert state.step == 1
    for name, update in updates.items():
        assert update.shape == params[name].shape
