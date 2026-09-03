"""Portable optimizer state, schedule, and validation contracts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from opaque.optimizers import (
    adadelta,
    adafactor,
    adagrad,
    adam,
    adamw,
    ademamix,
    apply_updates,
    lion,
    radam,
    rmsprop,
    schedule_free,
    sgd,
)
from opaque.serialization import from_state_dict, state_dict

_Factory = Callable[..., tuple[Callable[..., tuple[Any, Any]], Any]]
_FACTORIES: tuple[_Factory, ...] = (
    adadelta,
    adafactor,
    adagrad,
    adam,
    adamw,
    ademamix,
    lion,
    radam,
    rmsprop,
    sgd,
)


@pytest.mark.parametrize("factory", _FACTORIES, ids=lambda factory: factory.__name__)
def test_restored_optimizer_state_reproduces_the_next_update(
    backend_case, factory: _Factory
) -> None:
    params = {"weight": backend_case.array([1.0, -2.0])}
    first_grads = {"weight": backend_case.array([0.5, -0.25])}
    next_grads = {"weight": backend_case.array([0.2, 0.4])}
    kwargs = {"lr": 0.01} if factory is sgd else {}
    step, state = factory(params, **kwargs)
    _first_update, state = step(first_grads, state, params=params)
    restored = from_state_dict(state, state_dict(state))

    expected, expected_state = step(next_grads, state, params=params)
    actual, actual_state = step(next_grads, restored, params=params)

    backend_case.assert_allclose(actual["weight"], expected["weight"], rtol=0, atol=0)
    assert actual_state.step == expected_state.step == 2


def test_adam_alias_uses_l2_decay_and_rejects_invalid_configuration(
    backend_case,
) -> None:
    params = {"weight": backend_case.array([2.0, -1.0])}
    grads = {"weight": backend_case.array([0.5, -0.25])}
    adam_step, adam_state = adam(params, lr=0.1, weight_decay=0.5)
    l2_step, l2_state = adamw(
        params, lr=0.1, weight_decay=0.5, decoupled_weight_decay=False
    )

    adam_update, _ = adam_step(grads, adam_state, params=params)
    l2_update, _ = l2_step(grads, l2_state, params=params)

    backend_case.assert_allclose(adam_update["weight"], l2_update["weight"])
    with pytest.raises(ValueError, match="non-negative"):
        adamw(params, weight_decay=-1.0)
    with pytest.raises(ValueError, match="positive"):
        adamw(params, eps=0.0)
    with pytest.raises(ValueError, match="beta_1"):
        adamw(params, betas=(1.0, 0.999))
    with pytest.raises(ValueError, match="update_rms_clip"):
        adamw(params, update_rms_clip=0.0)


def test_schedule_free_tracks_the_raw_and_published_iterates_portably(
    backend_case,
) -> None:
    params = {"weight": backend_case.array([1.0, -1.0])}
    grads = {"weight": backend_case.array([0.5, -0.25])}
    step, state = schedule_free(params, sgd, lr=0.1, beta=0.5, warmup_steps=1)

    first_update, state = step(grads, state, params=params)
    first_params = apply_updates(params, first_update)
    backend_case.assert_allclose(state.x["weight"], state.z["weight"])

    second_update, state = step(grads, state, params=first_params)
    second_params = apply_updates(first_params, second_update)
    third_update, state = step(grads, state, params=second_params)
    third_params = apply_updates(second_params, third_update)
    assert state.step == 3
    assert not (
        backend_case.to_host(state.x["weight"])
        == backend_case.to_host(third_params["weight"])
    ).all()
    with pytest.raises(TypeError, match="params"):
        step(grads, state)
    with pytest.raises(ValueError, match="beta"):
        schedule_free(params, sgd, lr=0.1, beta=1.5)
