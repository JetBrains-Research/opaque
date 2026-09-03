"""Portable state and update contracts for every optimizer factory."""

from __future__ import annotations

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
from opaque.types import PerGroup, SecondMomentNoiseOutput, noised


@pytest.mark.parametrize(
    "factory",
    [adadelta, adafactor, adagrad, adam, adamw, ademamix, lion, radam, rmsprop, sgd],
    ids=lambda factory: factory.__name__,
)
def test_optimizer_factories_update_native_arrays_and_advance_state(
    backend_case, factory
) -> None:
    params = {
        "matrix": backend_case.array(
            [[1.0, -2.0], [0.5, 3.0]], dtype=backend_case.dtype("float32")
        )
    }
    grads = {
        "matrix": backend_case.array(
            [[0.25, -0.5], [1.0, -0.75]], dtype=backend_case.dtype("float32")
        )
    }
    kwargs = {"lr": 0.01} if factory is sgd else {}
    step, state = factory(params, **kwargs)

    updates, state = step(grads, state, params=params)
    next_params = apply_updates(params, updates)

    assert state.step == 1
    assert backend_case.to_host(updates["matrix"]).shape == (2, 2)
    assert backend_case.to_host(next_params["matrix"]).shape == (2, 2)


def test_optimizer_state_round_trips_with_native_array_leaves(backend_case) -> None:
    params = {"weight": backend_case.array([1.0, -2.0])}
    grads = {"weight": backend_case.array([0.5, -0.25])}
    step, state = adamw(params, lr=0.1)
    _updates, state = step(grads, state, params=params)

    restored = from_state_dict(state, state_dict(state))

    assert restored.step == state.step == 1
    backend_case.assert_allclose(restored.mu["weight"], state.mu["weight"])
    backend_case.assert_allclose(restored.nu["weight"], state.nu["weight"])


def test_schedule_free_wraps_a_portable_optimizer_and_tracks_averages(
    backend_case,
) -> None:
    params = {"weight": backend_case.array([1.0, -1.0])}
    grads = {"weight": backend_case.array([0.5, -0.25])}
    step, state = schedule_free(params, sgd, lr=0.1, beta=0.5)

    updates, state = step(grads, state, params=params)

    assert state.step == 1
    backend_case.assert_allclose(
        apply_updates(params, updates)["weight"],
        backend_case.to_host(state.z["weight"]) * 0.5
        + backend_case.to_host(state.x["weight"]) * 0.5,
    )


def test_adamw_distinguishes_decoupled_and_l2_weight_decay_portably(
    backend_case,
) -> None:
    params = {"weight": backend_case.array([2.0, -1.0])}
    grads = {"weight": backend_case.array([0.5, -0.25])}
    decoupled, decoupled_state = adamw(
        params, lr=0.1, weight_decay=0.5, decoupled_weight_decay=True
    )
    l2, l2_state = adamw(params, lr=0.1, weight_decay=0.5, decoupled_weight_decay=False)

    decoupled_updates, decoupled_state = decoupled(
        grads, decoupled_state, params=params
    )
    l2_updates, l2_state = l2(grads, l2_state, params=params)

    backend_case.assert_allclose(decoupled_updates["weight"], [-0.2, 0.15])
    assert not (
        backend_case.to_host(decoupled_state.mu["weight"])
        == backend_case.to_host(l2_state.mu["weight"])
    ).all()
    assert backend_case.to_host(l2_updates["weight"]).shape == (2,)


def test_adamw_noise_bias_correction_tracks_scalar_and_per_group_variance(
    backend_case,
) -> None:
    params = {
        "left": backend_case.array([1.0]),
        "right": backend_case.array([2.0]),
    }
    grads = {name: backend_case.array([0.5]) for name in params}
    step, state = adamw(params, lr=0.01, betas=(0.9, 0.5), noise_bias_correction=True)
    stddev = PerGroup(
        groups={"left": "small", "right": "large"},
        values={"small": 0.2, "large": 0.4},
    )

    for _ in range(2):
        _updates, state = step(
            noised(grads, max_norm=1.0, noise_stddev=stddev), state, params=params
        )

    assert state.phi[("left",)] == pytest.approx(0.03)
    assert state.phi[("right",)] == pytest.approx(0.12)


def test_adamw_second_moment_stream_and_rms_clip_are_portable(backend_case) -> None:
    params = {
        "big": backend_case.array([0.0, 0.0]),
        "small": backend_case.array([0.0, 0.0]),
    }
    grads = {
        "big": backend_case.array([10.0, -10.0]),
        "small": backend_case.array([1.0, -1.0]),
    }
    second_moment = {name: backend_case.array([100.0, 100.0]) for name in params}
    clipped_step, clipped_state = adamw(
        params, lr=1.0, weight_decay=0.0, update_rms_clip=0.1
    )
    reference_step, reference_state = adamw(params, lr=1.0, weight_decay=0.0)
    stream = SecondMomentNoiseOutput(
        noised(grads, max_norm=1.0, noise_stddev=0.1),
        noised(second_moment, max_norm=1.0, noise_stddev=0.1),
    )

    clipped_updates, _ = clipped_step(stream, clipped_state, params=params)
    reference_updates, _ = reference_step(stream, reference_state, params=params)
    clipped_host = backend_case.to_host(clipped_updates["big"])
    reference_host = backend_case.to_host(reference_updates["big"])

    assert 0.0 < abs(clipped_host[0] / reference_host[0]) < 1.0
    backend_case.assert_allclose(
        clipped_updates["small"],
        backend_case.to_host(reference_updates["small"])
        * abs(clipped_host[0] / reference_host[0]),
    )


def test_optimizer_schedules_and_warmup_advance_state_portably(backend_case) -> None:
    params = {"weight": backend_case.array([1.0, -2.0])}
    grads = {"weight": backend_case.array([0.5, -0.25])}
    calls: list[int] = []

    def schedule(step: int) -> float:
        calls.append(step)
        return 0.1 * step

    step, state = adamw(params, lr=schedule, weight_decay=0.1)
    first, state = step(grads, state, params=params)
    second, state = step(grads, state, params=params)

    backend_case.assert_allclose(first["weight"], [0.0, 0.0])
    assert abs(backend_case.to_host(second["weight"])).max() > 0.0
    assert calls == [0, 1]
    assert state.step == 2


def test_adamw_converges_on_a_quadratic_through_portable_updates(backend_case) -> None:
    target = backend_case.array([1.0, -2.0])
    params = {"weight": backend_case.array([0.0, 0.0])}
    step, state = adamw(params, lr=0.05, weight_decay=0.0)

    for _ in range(160):
        grads = {"weight": 2.0 * (params["weight"] - target)}
        updates, state = step(grads, state, params=params)
        params = apply_updates(params, updates)

    backend_case.assert_allclose(params["weight"], [1.0, -2.0], atol=0.08, rtol=0)


def test_restored_adam_state_produces_the_same_next_update(backend_case) -> None:
    params = {"weight": backend_case.array([1.0, -2.0])}
    first_grads = {"weight": backend_case.array([0.5, -0.25])}
    next_grads = {"weight": backend_case.array([0.2, 0.4])}
    step, state = adamw(params, lr=0.01, noise_bias_correction=True)
    _updates, state = step(
        noised(first_grads, max_norm=1.0, noise_stddev=0.3), state, params=params
    )
    restored = from_state_dict(state, state_dict(state))

    expected, expected_state = step(next_grads, state, params=params)
    actual, actual_state = step(next_grads, restored, params=params)

    backend_case.assert_allclose(actual["weight"], expected["weight"], rtol=0, atol=0)
    assert actual_state.step == expected_state.step == 2


@pytest.mark.parametrize(
    ("factory", "kwargs", "message"),
    [
        (adamw, {"weight_decay": -1.0}, "non-negative"),
        (adamw, {"eps": 0.0}, "positive"),
    ],
)
def test_optimizer_parameter_validation_is_provider_neutral(
    backend_case, factory, kwargs, message
) -> None:
    params = {"weight": backend_case.array([1.0])}

    with pytest.raises(ValueError, match=message):
        factory(params, **kwargs)
