"""Provider-neutral DP-SGD Gaussian-noise contracts."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.stats

from opaque import ops
from opaque.dpsgd.noise import gaussian_noise
from opaque.random import key
from opaque.serialization import from_state_dict, state_dict
from opaque.types import (
    ClippedPytree,
    NoisedPytree,
    PerGroup,
    SecondMomentClippingOutput,
    SecondMomentNoiseOutput,
    clipped,
    noised,
)


def _zeros(backend_case, shape: tuple[int, ...], dtype: str = "float32"):
    return backend_case.array(np.zeros(shape), dtype=backend_case.dtype(dtype))


def _host(backend_case, value) -> np.ndarray:
    return np.asarray(backend_case.to_host(value))


def test_gaussian_noise_preserves_pytree_metadata_and_advances_state(
    backend_case,
) -> None:
    gradients = {
        "weight": _zeros(backend_case, (32, 4)),
        "bias": _zeros(backend_case, (4,)),
    }
    noise_fn, state = gaussian_noise(noise_multiplier=0.5, key=key(7))

    output, state = noise_fn(clipped(gradients, max_norm=2.0), state)

    assert isinstance(output, NoisedPytree)
    assert output.max_norm == 2.0
    assert output.noise_stddev == 1.0
    assert state._step_counter == 1
    assert output.pytree["weight"].shape == (32, 4)
    assert output.pytree["weight"].dtype == backend_case.dtype("float32")
    assert not np.array_equal(_host(backend_case, output.pytree["weight"]), 0.0)


def test_gaussian_noise_is_keyed_and_checkpointable(backend_case) -> None:
    gradients = clipped({"w": _zeros(backend_case, (64,))}, max_norm=1.0)
    noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(17))
    first, state = noise_fn(gradients, state)
    checkpoint = state_dict(state)
    uninterrupted, uninterrupted_state = noise_fn(gradients, state)

    replay_fn, template = gaussian_noise(noise_multiplier=1.0, key=key(999))
    del replay_fn
    restored = from_state_dict(template, checkpoint)
    resumed, resumed_state = noise_fn(gradients, restored)

    assert not np.array_equal(_host(backend_case, first.pytree["w"]), 0.0)
    np.testing.assert_array_equal(
        _host(backend_case, uninterrupted.pytree["w"]),
        _host(backend_case, resumed.pytree["w"]),
    )
    assert uninterrupted_state._step_counter == resumed_state._step_counter == 2


@pytest.mark.parametrize("bound", [1.5, (-1.0, 2.0), [-0.5, 0.75]])
def test_bounded_gaussian_noise_stays_within_public_bound(backend_case, bound) -> None:
    low, high = (-bound, bound) if isinstance(bound, float) else bound
    noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=bound, key=key(23))
    output, _ = noise_fn(clipped(_zeros(backend_case, (2_048,)), max_norm=1.0), state)

    observed = _host(backend_case, output.pytree)
    assert observed.min() >= low - 1e-5
    assert observed.max() <= high + 1e-5


def test_zero_noise_is_identity_or_bounded_clamp(backend_case) -> None:
    values = backend_case.array([2.0, -2.0, 0.5], dtype=backend_case.dtype("float32"))
    identity_fn, identity_state = gaussian_noise(noise_multiplier=0.0, key=key(3))
    identity, _ = identity_fn(clipped(values, max_norm=math.inf), identity_state)
    bounded_fn, bounded_state = gaussian_noise(
        noise_multiplier=0.0, bound=1.0, key=key(3)
    )
    bounded, _ = bounded_fn(clipped(values, max_norm=1.0), bounded_state)

    np.testing.assert_array_equal(
        _host(backend_case, identity.pytree), _host(backend_case, values)
    )
    np.testing.assert_array_equal(
        _host(backend_case, bounded.pytree), np.array([1.0, -1.0, 0.5])
    )
    assert identity.noise_stddev == bounded.noise_stddev == 0.0


def test_gaussian_noise_validates_query_wrappers_and_multiplier(backend_case) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        gaussian_noise(noise_multiplier=-0.1, key=key(0))

    noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))
    values = _zeros(backend_case, (3,))
    with pytest.raises(TypeError, match="ClippedPytree"):
        noise_fn(values, state)
    with pytest.raises(TypeError, match="not NoisedPytree"):
        noise_fn(noised(values, max_norm=1.0, noise_stddev=1.0), state)


def test_gaussian_noise_supports_nested_and_tuple_pytrees(backend_case) -> None:
    nested = {
        "layer1": {
            "weight": _zeros(backend_case, (4, 2)),
            "bias": _zeros(backend_case, (2,)),
        },
        "layer2": (_zeros(backend_case, (3,)), _zeros(backend_case, (1,))),
    }
    noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(29))
    output, _ = noise_fn(clipped(nested, max_norm=1.0), state)

    assert set(output.pytree) == {"layer1", "layer2"}
    assert output.pytree["layer1"]["weight"].shape == (4, 2)
    assert len(output.pytree["layer2"]) == 2
    assert not np.array_equal(
        _host(backend_case, output.pytree["layer1"]["weight"]),
        _host(backend_case, nested["layer1"]["weight"]),
    )


def test_gaussian_noise_keyed_streams_are_replayable_and_independent(
    backend_case,
) -> None:
    gradients = clipped(_zeros(backend_case, (128,)), max_norm=1.0)
    first_fn, first_state = gaussian_noise(noise_multiplier=1.0, key=key(31))
    second_fn, second_state = gaussian_noise(noise_multiplier=1.0, key=key(31))
    other_fn, other_state = gaussian_noise(noise_multiplier=1.0, key=key(37))

    first, first_state = first_fn(gradients, first_state)
    replay, _ = second_fn(gradients, second_state)
    other, _ = other_fn(gradients, other_state)
    next_value, _ = first_fn(gradients, first_state)

    np.testing.assert_array_equal(
        _host(backend_case, first.pytree), _host(backend_case, replay.pytree)
    )
    assert not np.array_equal(
        _host(backend_case, first.pytree), _host(backend_case, other.pytree)
    )
    assert not np.array_equal(
        _host(backend_case, first.pytree), _host(backend_case, next_value.pytree)
    )


@pytest.mark.parametrize(
    "bound",
    [0.0, -1.0, (2.0, 1.0), (1.0, 2.0), (-2.0, -1.0), (1.0, 2.0, 3.0)],
)
def test_bounded_gaussian_noise_rejects_invalid_bounds(backend_case, bound) -> None:
    del backend_case
    with pytest.raises(ValueError, match="bound"):
        gaussian_noise(noise_multiplier=1.0, bound=bound, key=key(43))


def test_per_group_gaussian_zero_bounds_leave_the_group_unchanged(backend_case) -> None:
    maximum = PerGroup(
        groups={"noised": "noised", "clean": "clean"},
        values={"noised": 1.0, "clean": 0.0},
    )
    values = {
        "noised": backend_case.array([1.0] * 32, dtype=backend_case.dtype("float32")),
        "clean": backend_case.array([1.0] * 32, dtype=backend_case.dtype("float32")),
    }
    noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(47))
    output, _ = noise_fn(clipped(values, max_norm=maximum), state)

    np.testing.assert_array_equal(
        _host(backend_case, output.pytree["clean"]),
        _host(backend_case, values["clean"]),
    )
    assert not np.array_equal(
        _host(backend_case, output.pytree["noised"]),
        _host(backend_case, values["noised"]),
    )


def test_per_group_gaussian_noise_uses_independent_calibration(backend_case) -> None:
    maximum = PerGroup(
        groups={"small": "small", "large": "large"},
        values={"small": 1.0, "large": 4.0},
    )
    values = {
        "small": _zeros(backend_case, (4_096,)),
        "large": _zeros(backend_case, (4_096,)),
    }
    noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(41))
    output, state = noise_fn(clipped(values, max_norm=maximum), state)

    assert isinstance(output.noise_stddev, PerGroup)
    assert output.noise_stddev.values == {
        "small": pytest.approx(math.sqrt(5.0)),
        "large": pytest.approx(math.sqrt(20.0)),
    }
    assert state._step_counter == 1
    assert (
        _host(backend_case, output.pytree["large"]).var()
        > _host(backend_case, output.pytree["small"]).var() * 2.5
    )


def test_second_moment_gaussian_release_has_two_independent_streams(
    backend_case,
) -> None:
    values = {"w": _zeros(backend_case, (128,))}
    paired = SecondMomentClippingOutput(
        grads=clipped(values, max_norm=1.0),
        squared_grads=clipped(
            {"w": ops.multiply(values["w"], values["w"])}, max_norm=1.0
        ),
    )
    noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=4.0, key=key(53))
    output, state = noise_fn(paired, state)

    assert isinstance(output, SecondMomentNoiseOutput)
    assert output.noisy_grads.noise_stddev == pytest.approx(math.sqrt(2.0))
    assert output.noisy_squared_grads.noise_stddev == pytest.approx(math.sqrt(2.0))
    assert state._step_counter == 1
    assert not np.array_equal(
        _host(backend_case, output.noisy_grads.pytree["w"]),
        _host(backend_case, output.noisy_squared_grads.pytree["w"]),
    )


def test_gaussian_noise_preserves_shared_array_dtype(backend_case) -> None:
    values = _zeros(backend_case, (128,), "float32")
    noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(67))
    output, _ = noise_fn(clipped(values, max_norm=1.0), state)
    assert output.pytree.dtype == backend_case.dtype("float32")


def test_gaussian_noise_matches_standard_normal_distribution(backend_case) -> None:
    noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(42))
    output, _ = noise_fn(clipped(_zeros(backend_case, (10_000,)), max_norm=1.0), state)

    _, p_value = scipy.stats.kstest(
        _host(backend_case, output.pytree), scipy.stats.norm.cdf
    )
    assert p_value > 0.01, f"KS test failed with p={p_value}"


def test_gaussian_noise_measured_stddev_matches_target(backend_case) -> None:
    target_stddev = 2.5
    noise_fn, state = gaussian_noise(noise_multiplier=target_stddev, key=key(0))
    output, state = noise_fn(
        clipped(_zeros(backend_case, (10_000,)), max_norm=1.0), state
    )

    assert output.noise_stddev == pytest.approx(target_stddev)
    measured = float(_host(backend_case, output.pytree).std())
    assert abs(measured - target_stddev) < 0.1

    clipped_value = ClippedPytree(_zeros(backend_case, (10_000,)), max_norm=2.0)
    clipped_fn, clipped_state = gaussian_noise(noise_multiplier=1.5, key=key(0))
    clipped_output, _ = clipped_fn(clipped_value, clipped_state)
    assert clipped_output.noise_stddev == pytest.approx(3.0)
    clipped_measured = float(_host(backend_case, clipped_output.pytree).std())
    assert abs(clipped_measured - 3.0) < 0.1


def test_gaussian_noise_requires_multiplier_and_key(backend_case) -> None:
    del backend_case
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument"):
        gaussian_noise(key=key(0))
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument"):
        gaussian_noise(noise_multiplier=1.0)
