"""Portable rule/state/update contracts for non-Adam optimizer families."""

from __future__ import annotations

import math

import pytest

from opaque.optimizers import (
    adadelta,
    adafactor,
    adagrad,
    adamw,
    ademamix,
    apply_updates,
    lion,
    radam,
    rmsprop,
    sgd,
)
from opaque.optimizers.types import AdEMAMixState, LionState, RAdamState, SGDState
from opaque.serialization import from_state_dict, state_dict
from opaque.types import PerGroup, SecondMomentNoiseOutput, clipped, noised


def _params(backend_case):
    return {
        "weight": backend_case.array(
            [[0.5, -1.0], [1.5, 0.25]], dtype=backend_case.dtype("float32")
        ),
        "bias": backend_case.array([0.1, -0.2], dtype=backend_case.dtype("float32")),
    }


def _grads(backend_case):
    return {
        "weight": backend_case.array(
            [[0.2, -0.4], [0.5, -0.1]], dtype=backend_case.dtype("float32")
        ),
        "bias": backend_case.array([0.3, -0.6], dtype=backend_case.dtype("float32")),
    }


def test_sgd_rejects_clipped_and_explicit_metadata_and_accepts_noised(
    backend_case,
) -> None:
    params = _params(backend_case)
    grads = _grads(backend_case)
    step, state = sgd(params, lr=1e-2)

    with pytest.raises(TypeError, match="have not passed through a noise mechanism"):
        step(clipped(grads, max_norm=1.0), state, params=params)
    with pytest.raises(TypeError, match="noise_stddev"):
        step(grads, state, params=params, noise_stddev=0.5)
    with pytest.raises(TypeError, match="noisy_squared_grads"):
        step(grads, state, params=params, noisy_squared_grads={})

    updates, _ = step(
        noised(grads, max_norm=1.0, noise_stddev=0.25), state, params=params
    )
    for name in params:
        assert (
            backend_case.to_host(updates[name]).shape
            == backend_case.to_host(params[name]).shape
        )

    sq = {name: backend_case.to_host(value) ** 2 for name, value in grads.items()}
    sq = {name: backend_case.array(value) for name, value in sq.items()}
    output = SecondMomentNoiseOutput(
        noised(grads, max_norm=1.0, noise_stddev=0.1),
        noised(sq, max_norm=1.0, noise_stddev=0.1),
    )
    updates, state = step(output, state, params=params)
    for name in params:
        assert (
            backend_case.to_host(updates[name]).shape
            == backend_case.to_host(params[name]).shape
        )


def test_sgd_schedules_warmup_and_state_round_trip(backend_case) -> None:
    params = {"w": backend_case.array([1.0, 1.0])}
    zero_grads = {"w": backend_case.array([0.0, 0.0])}
    calls: list[int] = []

    def schedule(step: int) -> float:
        calls.append(step)
        return 0.1 * (step + 1)

    step, state = sgd(params, lr=schedule, momentum=0.0, weight_decay=0.0)
    for _ in range(3):
        _, state = step(zero_grads, state, params=params)
    assert calls == [0, 1, 2]

    step, state = sgd(params, lr=schedule, momentum=0.0, weight_decay=0.1)
    expected_lrs = [0.1, 0.2, 0.3]
    # Fresh schedule counter for the weight-decay application check.
    calls.clear()
    for index in range(3):
        updates, state = step(zero_grads, state, params=params)
        backend_case.assert_allclose(
            updates["w"],
            backend_case.to_host(params["w"]) * (-expected_lrs[index] * 0.1),
        )

    def warmup(step_idx: int) -> float:
        return 1e-2 * min(step_idx / 2, 1.0)

    params = _params(backend_case)
    grads = _grads(backend_case)
    step, state = sgd(params, lr=warmup, momentum=0.9, weight_decay=0.01)
    first, state = step(grads, state, params=params)
    assert all(abs(backend_case.to_host(first[name])).sum() == 0.0 for name in first)
    second, state = step(grads, state, params=params)
    assert any(abs(backend_case.to_host(second[name])).sum() > 0.0 for name in second)

    _, state = step(grads, state, params=params)
    restored = from_state_dict(state, state_dict(state))
    assert isinstance(restored, SGDState)
    assert restored.step == state.step
    for name in params:
        backend_case.assert_allclose(restored.momentum[name], state.momentum[name])


def test_lion_signed_updates_weight_decay_and_validation(backend_case) -> None:
    params = _params(backend_case)
    grads = _grads(backend_case)
    step, state = lion(params, lr=1.0, weight_decay=0.0)
    assert isinstance(state, LionState)
    assert state.step == 0
    for name in params:
        backend_case.assert_allclose(
            state.m[name],
            backend_case.to_host(params[name]) * 0.0,
        )

    updates, state = step(grads, state, params=params)
    for name in updates:
        host = abs(backend_case.to_host(updates[name]))
        assert (host == 1.0).all()
    assert state.step == 1

    zero_params = {"w": backend_case.array([2.0, 2.0, 2.0])}
    zero_grads = {"w": backend_case.array([0.0, 0.0, 0.0])}
    step, state = lion(
        zero_params, lr=0.1, weight_decay=0.5, decoupled_weight_decay=True
    )
    updates, _ = step(zero_grads, state, params=zero_params)
    backend_case.assert_allclose(updates["w"], [-0.1, -0.1, -0.1])

    with pytest.raises(ValueError, match="betas must contain two values"):
        lion({"w": backend_case.array([1.0])}, betas=(1.0, 0.99))
    with pytest.raises(ValueError, match="weight_decay must be non-negative"):
        lion({"w": backend_case.array([1.0])}, weight_decay=-0.1)

    step, state = lion(params, lr=1e-4)
    with pytest.raises(TypeError, match="noise_stddev"):
        step(grads, state, params=params, noise_stddev=0.5)
    with pytest.raises(TypeError, match="have not passed through a noise mechanism"):
        step(clipped(grads, max_norm=1.0), state, params=params)
    updates, _ = step(
        noised(grads, max_norm=1.0, noise_stddev=0.25), state, params=params
    )
    for name in params:
        assert (
            backend_case.to_host(updates[name]).shape
            == backend_case.to_host(params[name]).shape
        )


def test_ademamix_moments_bc_second_moment_and_validation(backend_case) -> None:
    params = _params(backend_case)
    grads = _grads(backend_case)
    b1, b2, b3 = 0.9, 0.999, 0.9999
    step, state = ademamix(
        params, lr=1e-3, betas=(b1, b2, b3), alpha=5.0, weight_decay=0.0
    )
    assert isinstance(state, AdEMAMixState)
    _, state = step(grads, state, params=params)
    for name in grads:
        backend_case.assert_allclose(
            state.m_fast[name],
            backend_case.to_host(grads[name]) * (1 - b1),
        )
        backend_case.assert_allclose(
            state.m_slow[name],
            backend_case.to_host(grads[name]) * (1 - b3),
        )
        backend_case.assert_allclose(
            state.nu[name],
            backend_case.to_host(grads[name]) ** 2 * (1 - b2),
        )

    step_ame, s_ame = ademamix(
        params, lr=1e-3, betas=(0.9, 0.999, 0.9999), alpha=0.0, weight_decay=0.01
    )
    step_adam, s_adam = adamw(params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
    for _ in range(5):
        u_ame, s_ame = step_ame(grads, s_ame, params=params)
        u_adam, s_adam = step_adam(grads, s_adam, params=params)
        for name in params:
            backend_case.assert_allclose(u_ame[name], u_adam[name])

    sigma = 0.4
    step, state = ademamix(
        params, lr=1e-3, betas=(0.9, b2, 0.9999), noise_bias_correction=True
    )
    expected_phi = 0.0
    for _ in range(8):
        _, state = step(
            noised(grads, max_norm=1.0, noise_stddev=sigma), state, params=params
        )
        expected_phi = b2 * expected_phi + (1 - b2) * (sigma**2)
    assert isinstance(state.phi, dict)
    assert all(value == pytest.approx(expected_phi) for value in state.phi.values())

    step, state = ademamix(params, lr=1e-3, noise_bias_correction=False)
    _, state = step(noised(grads, max_norm=1.0, noise_stddev=0.4), state, params=params)
    assert state.phi == 0.0

    sq = {
        name: backend_case.array(backend_case.to_host(value) ** 2 + 0.05)
        for name, value in grads.items()
    }
    step, state = ademamix(params, lr=1e-3, betas=(0.9, b2, 0.9999))
    output = SecondMomentNoiseOutput(
        noised(grads, max_norm=1.0, noise_stddev=0.1),
        noised(sq, max_norm=1.0, noise_stddev=0.1),
    )
    _, state = step(output, state, params=params)
    for name in params:
        backend_case.assert_allclose(
            state.nu[name], backend_case.to_host(sq[name]) * (1 - b2)
        )

    negative = {
        name: backend_case.array(-1.0 * (backend_case.to_host(value) * 0 + 1))
        for name, value in grads.items()
    }
    step, state = ademamix(params, lr=1e-3)
    output = SecondMomentNoiseOutput(
        noised(grads, max_norm=1.0, noise_stddev=0.1),
        noised(negative, max_norm=1.0, noise_stddev=0.1),
    )
    updates, _ = step(output, state, params=params)
    for name in params:
        host = backend_case.to_host(updates[name])
        assert (abs(host) < 10.0).all()

    step, state = ademamix(params, lr=1e-3)
    with pytest.raises(TypeError, match="noisy_squared_grads"):
        step(grads, state, params=params, noisy_squared_grads=grads)
    with pytest.raises(ValueError, match=r"invalid AdEMAMix|betas"):
        ademamix(params, betas=(0.9, 0.999))
    with pytest.raises(ValueError, match=r"invalid AdEMAMix|alpha"):
        ademamix(params, alpha=-1.0)


def test_radam_rectification_bc_second_moment_and_validation(backend_case) -> None:
    from opaque.api.optimizers._radam import _rectification, _rho_t

    assert _rho_t(0.999, 10**9) == pytest.approx(2.0 / (1.0 - 0.999) - 1.0)
    assert _rectification(0.999, 1) is None
    assert _rectification(0.999, 10**6) is not None

    params = _params(backend_case)
    grads = _grads(backend_case)
    step, state = radam(params, lr=1e-3)
    assert isinstance(state, RAdamState)
    assert state.step == 0
    for _ in range(6):
        updates, state = step(grads, state, params=params)
        for name in updates:
            assert (
                backend_case.to_host(updates[name]).shape
                == backend_case.to_host(params[name]).shape
            )
    assert state.step == 6

    b2 = 0.999
    sigma = 0.3
    step, state = radam(params, lr=1e-3, betas=(0.9, b2), noise_bias_correction=True)
    expected_phi = 0.0
    for _ in range(4):
        _, state = step(
            noised(grads, max_norm=1.0, noise_stddev=sigma), state, params=params
        )
        expected_phi = b2 * expected_phi + (1 - b2) * (sigma**2)
    assert isinstance(state.phi, dict)
    assert all(value == pytest.approx(expected_phi) for value in state.phi.values())

    step, state = radam(params, lr=1e-3, noise_bias_correction=False)
    _, state = step(noised(grads, max_norm=1.0, noise_stddev=0.4), state, params=params)
    assert state.phi == 0.0
    step, state = radam(params, lr=1e-3, noise_bias_correction=True)
    _, state = step(grads, state, params=params)
    assert isinstance(state.phi, dict)
    assert all(value == 0.0 for value in state.phi.values())

    step_on, state_on = radam(
        params, lr=1e-2, betas=(0.9, 0.999), noise_bias_correction=True
    )
    step_off, state_off = radam(
        params, lr=1e-2, betas=(0.9, 0.999), noise_bias_correction=False
    )
    for _ in range(20):
        u_on, state_on = step_on(
            noised(grads, max_norm=1.0, noise_stddev=0.5), state_on, params=params
        )
        u_off, state_off = step_off(
            noised(grads, max_norm=1.0, noise_stddev=0.5), state_off, params=params
        )
    assert not all(
        (backend_case.to_host(u_on[name]) == backend_case.to_host(u_off[name])).all()
        for name in params
    )

    step, state = radam(params, lr=1e-3, noise_bias_correction=True)
    stddev = PerGroup(
        groups={"weight": "w", "bias": "b"},
        values={"w": 0.2, "b": 0.5},
    )
    for _ in range(3):
        _, state = step(
            noised(grads, max_norm=1.0, noise_stddev=stddev), state, params=params
        )
    assert isinstance(state.phi, dict)
    assert state.phi[("weight",)] != state.phi[("bias",)]

    sq = {
        name: backend_case.array(abs(backend_case.to_host(value)) + 0.1)
        for name, value in grads.items()
    }
    step, state = radam(params, lr=1e-3, noise_bias_correction=True)
    output = SecondMomentNoiseOutput(
        noised(grads, max_norm=1.0, noise_stddev=0.1),
        noised(sq, max_norm=1.0, noise_stddev=0.1),
    )
    _, state = step(output, state, params=params)
    assert isinstance(state.phi, dict)
    assert all(value == 0.0 for value in state.phi.values())

    with pytest.raises(ValueError, match="invalid RAdam"):
        radam(params, eps=0.0)
    with pytest.raises(ValueError, match="invalid RAdam"):
        radam(params, betas=(0.9,))
    with pytest.raises(ValueError, match="invalid RAdam"):
        radam(params, betas=(1.0, 0.999))
    with pytest.raises(ValueError, match="invalid RAdam"):
        radam(params, weight_decay=-0.1)
    with pytest.raises(ValueError, match="update_rms_clip must be positive"):
        radam(params, update_rms_clip=0.0)


def test_adagrad_accumulator_bc_weight_decay_and_validation(backend_case) -> None:
    params = _params(backend_case)
    grads = _grads(backend_case)

    step, state = adagrad(params, lr=1e-2, initial_accumulator_value=0.1)
    for name in params:
        backend_case.assert_allclose(
            state.v_acc[name],
            backend_case.to_host(params[name]) * 0.0 + 0.1,
        )
    _, state = step(grads, state, params=params)
    for name in params:
        expected = 0.1 + backend_case.to_host(grads[name]) ** 2
        backend_case.assert_allclose(state.v_acc[name], expected)
    assert state.phi_acc == 0.0

    sigma = 0.4
    step, state = adagrad(params, lr=1e-2, noise_bias_correction=True)
    for index in range(5):
        _, state = step(
            noised(grads, max_norm=1.0, noise_stddev=sigma), state, params=params
        )
        expected = (index + 1) * sigma**2
        if isinstance(state.phi_acc, dict):
            assert all(
                value == pytest.approx(expected) for value in state.phi_acc.values()
            )
        else:
            assert state.phi_acc == pytest.approx(expected)

    step, state = adagrad(params, lr=1e-2, noise_bias_correction=False)
    _, state = step(noised(grads, max_norm=1.0, noise_stddev=0.4), state, params=params)
    assert state.phi_acc == 0.0

    # Correction removes additive noise variance from the denominator.
    params_1d = {"w": backend_case.array([0.0])}
    grads_1d = {"w": backend_case.array([1.0])}
    g = 1.0
    sigma = 0.5
    steps = 5
    lr = 1e-2
    eps = 1e-8
    step_bc, state_bc = adagrad(
        params_1d, lr=lr, eps=eps, weight_decay=0.0, noise_bias_correction=True
    )
    step_off, state_off = adagrad(
        params_1d, lr=lr, eps=eps, weight_decay=0.0, noise_bias_correction=False
    )
    for _ in range(steps):
        stream = noised(grads_1d, max_norm=1.0, noise_stddev=sigma)
        u_bc, state_bc = step_bc(stream, state_bc, params=params_1d)
        u_off, state_off = step_off(stream, state_off, params=params_1d)
    expected_gain = math.sqrt(g**2 / (g**2 - sigma**2))
    gain = float(
        backend_case.to_host(u_bc["w"])[0] / backend_case.to_host(u_off["w"])[0]
    )
    assert gain == pytest.approx(expected_gain, rel=1e-5)

    # Noise-dominant regime falls back to the uncorrected accumulator.
    huge_sigma = 1e3
    step_bc, state_bc = adagrad(
        params_1d, lr=lr, eps=eps, weight_decay=0.0, noise_bias_correction=True
    )
    step_raw, state_raw = adagrad(
        params_1d, lr=lr, eps=eps, weight_decay=0.0, noise_bias_correction=False
    )
    for _ in range(5):
        stream = noised(grads_1d, max_norm=1.0, noise_stddev=huge_sigma)
        u_bc, state_bc = step_bc(stream, state_bc, params=params_1d)
        u_raw, state_raw = step_raw(stream, state_raw, params=params_1d)
    backend_case.assert_allclose(u_bc["w"], u_raw["w"], rtol=0, atol=0)

    zero_params = {"w": backend_case.array([2.0, 2.0])}
    zero_grads = {"w": backend_case.array([0.0, 0.0])}
    step, state = adagrad(
        zero_params, lr=0.1, weight_decay=0.5, decoupled_weight_decay=True
    )
    updates, _ = step(zero_grads, state, params=zero_params)
    backend_case.assert_allclose(updates["w"], [-0.1, -0.1])

    with pytest.raises(ValueError, match="invalid Adagrad"):
        adagrad(params, eps=-1e-8)
    with pytest.raises(ValueError, match="invalid Adagrad"):
        adagrad(params, initial_accumulator_value=-0.1)
    with pytest.raises(ValueError, match="invalid Adagrad"):
        adagrad(params, weight_decay=-0.1)


def test_rmsprop_ema_bc_second_moment_and_validation(backend_case) -> None:
    params = _params(backend_case)
    grads = _grads(backend_case)
    alpha = 0.99
    step, state = rmsprop(params, lr=1e-2, alpha=alpha)
    assert state.phi == 0.0
    _, state = step(grads, state, params=params)
    for name in params:
        backend_case.assert_allclose(
            state.nu[name],
            backend_case.to_host(grads[name]) ** 2 * (1 - alpha),
        )
    assert state.step == 1
    next_params = apply_updates(params, step(grads, state, params=params)[0])
    assert any(
        not (
            backend_case.to_host(next_params[name])
            == backend_case.to_host(params[name])
        ).all()
        for name in params
    )

    b2 = alpha
    sigma = 0.4
    step, state = rmsprop(params, lr=1e-2, alpha=b2, noise_bias_correction=True)
    expected_phi = 0.0
    for _ in range(6):
        _, state = step(
            noised(grads, max_norm=1.0, noise_stddev=sigma), state, params=params
        )
        expected_phi = b2 * expected_phi + (1 - b2) * (sigma**2)
    if isinstance(state.phi, dict):
        assert all(value == pytest.approx(expected_phi) for value in state.phi.values())
    else:
        assert state.phi == pytest.approx(expected_phi)

    step, state = rmsprop(params, lr=1e-2, noise_bias_correction=False)
    _, state = step(noised(grads, max_norm=1.0, noise_stddev=0.4), state, params=params)
    assert state.phi == 0.0

    params_1d = {"w": backend_case.array([0.0])}
    grads_1d = {"w": backend_case.array([1.0])}
    g = 1.0
    sigma = 0.5
    alpha = 0.9
    lr = 1e-2
    eps = 1e-8
    steps = 8
    step_bc, state_bc = rmsprop(
        params_1d,
        lr=lr,
        alpha=alpha,
        eps=eps,
        weight_decay=0.0,
        noise_bias_correction=True,
    )
    step_off, state_off = rmsprop(
        params_1d,
        lr=lr,
        alpha=alpha,
        eps=eps,
        weight_decay=0.0,
        noise_bias_correction=False,
    )
    for _ in range(steps):
        stream = noised(grads_1d, max_norm=1.0, noise_stddev=sigma)
        u_bc, state_bc = step_bc(stream, state_bc, params=params_1d)
        u_off, state_off = step_off(stream, state_off, params=params_1d)
    bc = 1.0 - alpha**steps
    expected_on = -lr * g / (math.sqrt(bc * (g**2 - sigma**2)) + eps)
    expected_off = -lr * g / (math.sqrt(bc * g**2) + eps)
    backend_case.assert_allclose(u_bc["w"], [expected_on], rtol=1e-5, atol=0)
    backend_case.assert_allclose(u_off["w"], [expected_off], rtol=1e-5, atol=0)

    step_bc, state_bc = rmsprop(
        params_1d,
        lr=lr,
        alpha=alpha,
        eps=eps,
        weight_decay=0.0,
        noise_bias_correction=True,
    )
    step_raw, state_raw = rmsprop(
        params_1d,
        lr=lr,
        alpha=alpha,
        eps=eps,
        weight_decay=0.0,
        noise_bias_correction=False,
    )
    for _ in range(5):
        stream = noised(grads_1d, max_norm=1.0, noise_stddev=1e6)
        u_bc, state_bc = step_bc(stream, state_bc, params=params_1d)
        u_raw, state_raw = step_raw(stream, state_raw, params=params_1d)
    backend_case.assert_allclose(u_bc["w"], u_raw["w"], rtol=0, atol=0)

    sq = {
        name: backend_case.array(backend_case.to_host(value) ** 2 + 0.05)
        for name, value in grads.items()
    }
    step, state = rmsprop(params, lr=1e-2, alpha=alpha)
    output = SecondMomentNoiseOutput(
        noised(grads, max_norm=1.0, noise_stddev=0.1),
        noised(sq, max_norm=1.0, noise_stddev=0.1),
    )
    _, state = step(output, state, params=params)
    for name in params:
        backend_case.assert_allclose(
            state.nu[name], backend_case.to_host(sq[name]) * (1 - alpha)
        )

    negative = {
        name: backend_case.array(-(backend_case.to_host(value) * 0 + 1))
        for name, value in grads.items()
    }
    step, state = rmsprop(params, lr=1e-2)
    output = SecondMomentNoiseOutput(
        noised(grads, max_norm=1.0, noise_stddev=0.1),
        noised(negative, max_norm=1.0, noise_stddev=0.1),
    )
    updates, _ = step(output, state, params=params)
    for name in params:
        assert (abs(backend_case.to_host(updates[name])) < 10.0).all()

    step, state = rmsprop(params, lr=1e-2)
    with pytest.raises(TypeError, match="noisy_squared_grads"):
        step(grads, state, params=params, noisy_squared_grads=sq)
    with pytest.raises(ValueError, match="invalid RMSprop"):
        rmsprop(params, alpha=-0.1)
    with pytest.raises(ValueError, match="invalid RMSprop"):
        rmsprop(params, alpha=1.0)
    with pytest.raises(ValueError, match="invalid RMSprop"):
        rmsprop(params, eps=-1e-8)


def test_adadelta_state_bc_second_moment_and_validation(backend_case) -> None:
    params = _params(backend_case)
    grads = _grads(backend_case)
    rho = 0.9
    step, state = adadelta(params, lr=1.0, rho=rho)
    assert state.step == 0
    _, state = step(grads, state, params=params)
    assert state.step == 1
    for name in params:
        backend_case.assert_allclose(
            state.v_g[name],
            backend_case.to_host(grads[name]) ** 2 * (1 - rho),
        )
        assert (backend_case.to_host(state.v_dx[name]) >= 0).all()

    step, state = adadelta(params, lr=1.0, rho=rho, noise_bias_correction=True)
    sigma = 0.3
    expected = 0.0
    for _ in range(8):
        _, state = step(
            noised(grads, max_norm=1.0, noise_stddev=sigma), state, params=params
        )
        expected = rho * expected + (1 - rho) * (sigma**2)
    if isinstance(state.phi_g, dict):
        assert all(value == pytest.approx(expected) for value in state.phi_g.values())
    else:
        assert state.phi_g == pytest.approx(expected)
    for name in params:
        assert (backend_case.to_host(state.phi_dx[name]) >= 0).all()

    step, state = adadelta(params, lr=1.0, noise_bias_correction=False)
    _, state = step(noised(grads, max_norm=1.0, noise_stddev=0.4), state, params=params)
    assert state.phi_g is None
    assert state.phi_dx is None

    step, state = adadelta(params, lr=1.0, noise_bias_correction=True)
    _, state = step(grads, state, params=params)
    if isinstance(state.phi_g, dict):
        assert all(value == 0.0 for value in state.phi_g.values())
    else:
        assert state.phi_g == 0.0

    step_on, s_on = adadelta(params, lr=1.0, rho=0.9, noise_bias_correction=True)
    step_off, s_off = adadelta(params, lr=1.0, rho=0.9, noise_bias_correction=False)
    for _ in range(10):
        u_on, s_on = step_on(
            noised(grads, max_norm=1.0, noise_stddev=0.5), s_on, params=params
        )
        u_off, s_off = step_off(
            noised(grads, max_norm=1.0, noise_stddev=0.5), s_off, params=params
        )
    assert not all(
        (backend_case.to_host(u_on[name]) == backend_case.to_host(u_off[name])).all()
        for name in params
    )

    stddev = PerGroup(
        groups={"weight": "w", "bias": "b"},
        values={"w": 0.2, "b": 0.5},
    )
    step, state = adadelta(params, lr=1.0, rho=0.9, noise_bias_correction=True)
    for _ in range(3):
        _, state = step(
            noised(grads, max_norm=1.0, noise_stddev=stddev), state, params=params
        )
    assert isinstance(state.phi_g, dict)
    assert state.phi_g[("weight",)] != state.phi_g[("bias",)]

    sq = {
        name: backend_case.array(backend_case.to_host(value) ** 2 + 0.05)
        for name, value in grads.items()
    }
    step, state = adadelta(params, lr=1.0, rho=rho, noise_bias_correction=True)
    output = SecondMomentNoiseOutput(
        noised(grads, max_norm=1.0, noise_stddev=0.1),
        noised(sq, max_norm=1.0, noise_stddev=0.1),
    )
    _, state = step(output, state, params=params)
    if isinstance(state.phi_g, dict):
        assert all(value == 0.0 for value in state.phi_g.values())
    else:
        assert state.phi_g == 0.0

    negative = {
        name: backend_case.array(-(backend_case.to_host(value) * 0 + 1))
        for name, value in grads.items()
    }
    step, state = adadelta(params, lr=1.0)
    output = SecondMomentNoiseOutput(
        noised(grads, max_norm=1.0, noise_stddev=0.1),
        noised(negative, max_norm=1.0, noise_stddev=0.1),
    )
    updates, _ = step(output, state, params=params)
    for name in params:
        assert (abs(backend_case.to_host(updates[name])) < 10.0).all()

    with pytest.raises(ValueError, match="invalid Adadelta"):
        adadelta(params, eps=0.0)
    with pytest.raises(ValueError, match="invalid Adadelta"):
        adadelta(params, rho=1.0)
    with pytest.raises(ValueError, match="invalid Adadelta"):
        adadelta(params, weight_decay=-0.1)
    with pytest.raises(ValueError, match="invalid Adadelta"):
        adadelta(params, update_rms_clip=0.0)


def test_adafactor_factored_state_bc_clip_and_validation(backend_case) -> None:
    matrix_params = {
        "matrix": backend_case.array(
            [[1.0, -2.0, 0.5], [0.25, 1.5, -0.75]],
            dtype=backend_case.dtype("float32"),
        ),
        "vector": backend_case.array([0.5, -1.0], dtype=backend_case.dtype("float32")),
    }
    matrix_grads = {
        "matrix": backend_case.array(
            [[0.2, -0.1, 0.4], [-0.3, 0.5, 0.1]],
            dtype=backend_case.dtype("float32"),
        ),
        "vector": backend_case.array([0.25, -0.5], dtype=backend_case.dtype("float32")),
    }

    step, state = adafactor(matrix_params, lr=1e-2)
    # Factored second moments for matrices; unfactored for vectors.
    assert len(state.v_flat) == 2
    lengths = sorted(len(entry) for entry in state.v_flat)
    assert lengths == [1, 2]
    matrix_entry = next(entry for entry in state.v_flat if len(entry) == 2)
    vector_entry = next(entry for entry in state.v_flat if len(entry) == 1)
    assert backend_case.to_host(matrix_entry[0]).shape == (2,)
    assert backend_case.to_host(matrix_entry[1]).shape == (3,)
    assert backend_case.to_host(vector_entry[0]).shape == (2,)

    _, state = step(matrix_grads, state, params=matrix_params)
    assert state.step == 1
    updates, _ = step(matrix_grads, state, params=matrix_params)
    changed = apply_updates(matrix_params, updates)
    assert not (
        backend_case.to_host(changed["matrix"])
        == backend_case.to_host(matrix_params["matrix"])
    ).all()

    step, state = adafactor(matrix_params, lr=1e-2, beta1=0.0)
    assert state.m is None
    step, state = adafactor(matrix_params, lr=1e-2, beta1=0.9)
    assert state.m is not None

    zero_params = {"w": backend_case.array([2.0, 2.0])}
    zero_grads = {"w": backend_case.array([0.0, 0.0])}
    step, state = adafactor(
        zero_params, lr=0.1, weight_decay=0.5, decoupled_weight_decay=True
    )
    updates, _ = step(zero_grads, state, params=zero_params)
    backend_case.assert_allclose(updates["w"], [-0.1, -0.1])

    # Global RMS clip uses one scale across leaves.
    params = {
        "big": backend_case.array([0.0, 0.0]),
        "small": backend_case.array([0.0, 0.0]),
    }
    grads = {
        "big": backend_case.array([10.0, -10.0]),
        "small": backend_case.array([1.0, -1.0]),
    }
    clipped_step, clipped_state = adafactor(
        params, lr=1.0, weight_decay=0.0, beta1=0.0, update_rms_clip=0.1
    )
    reference_step, reference_state = adafactor(
        params, lr=1.0, weight_decay=0.0, beta1=0.0, update_rms_clip=1e6
    )
    clipped_updates, _ = clipped_step(grads, clipped_state, params=params)
    reference_updates, _ = reference_step(grads, reference_state, params=params)
    scale = abs(backend_case.to_host(clipped_updates["big"])[0]) / abs(
        backend_case.to_host(reference_updates["big"])[0]
    )
    assert 0.0 < scale < 1.0
    backend_case.assert_allclose(
        clipped_updates["small"],
        backend_case.to_host(reference_updates["small"]) * scale,
        rtol=1e-5,
        atol=1e-5,
    )

    step, state = adafactor(matrix_params, lr=1e-2)
    with pytest.raises(TypeError, match="noise_stddev"):
        step(matrix_grads, state, params=matrix_params, noise_stddev=0.5)
    with pytest.raises(TypeError, match="noisy_squared_grads"):
        step(matrix_grads, state, params=matrix_params, noisy_squared_grads={})

    step, state = adafactor(matrix_params, lr=1e-2, noise_bias_correction=True)
    sigma = 0.3
    expected = 0.0
    decay_rate = -0.8
    for step_idx in range(1, 5):
        _, state = step(
            noised(matrix_grads, max_norm=1.0, noise_stddev=sigma),
            state,
            params=matrix_params,
        )
        beta2t = 1.0 - math.pow(step_idx, decay_rate)
        expected = beta2t * expected + (1.0 - beta2t) * (sigma**2)
    assert all(value == pytest.approx(expected, rel=1e-5) for value in state.phi_flat)

    step, state = adafactor(matrix_params, lr=1e-2, noise_bias_correction=False)
    _, state = step(
        noised(matrix_grads, max_norm=1.0, noise_stddev=0.4),
        state,
        params=matrix_params,
    )
    assert all(value == 0.0 for value in state.phi_flat)

    stddev = PerGroup(
        groups={"matrix": "m", "vector": "v"},
        values={"m": 0.2, "v": 0.5},
    )
    step, state = adafactor(matrix_params, lr=1e-2, noise_bias_correction=True)
    for _ in range(3):
        _, state = step(
            noised(matrix_grads, max_norm=1.0, noise_stddev=stddev),
            state,
            params=matrix_params,
        )
    assert len(set(state.phi_flat)) > 1

    with pytest.raises(ValueError, match="invalid Adafactor"):
        adafactor(matrix_params, decay_rate=0.1)
    with pytest.raises(ValueError, match="invalid Adafactor"):
        adafactor(matrix_params, eps_root=0.0)
