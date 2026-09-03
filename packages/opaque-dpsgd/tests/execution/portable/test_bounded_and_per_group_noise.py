"""Provider-neutral bounded and per-group Gaussian-noise behavior."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.stats

from opaque.dpsgd.clipping import clipped_grad, per_group
from opaque.dpsgd.noise import gaussian_noise
from opaque.dpsgd.noise.types import GaussianNoiseState
from opaque.random import key
from opaque.types import NoisedPytree, PerGroup, clipped, noised


def _array(backend_case, value):
    return backend_case.array(value, dtype=backend_case.dtype("float32"))


def _host(backend_case, value):
    return np.asarray(backend_case.to_host(value))


def test_bounded_noise_preserves_metadata_and_pytree_shape(backend_case) -> None:
    gradients = {
        "weight": _array(backend_case, np.zeros((10, 5))),
        "bias": _array(backend_case, np.zeros(10)),
    }
    noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=5.0, key=key(0))
    output, state = noise_fn(clipped(gradients, max_norm=1.0), state)

    assert isinstance(state, GaussianNoiseState)
    assert isinstance(output, NoisedPytree)
    assert set(output.pytree) == set(gradients)
    assert output.pytree["weight"].shape == (10, 5)
    assert output.pytree["weight"].dtype == backend_case.dtype("float32")
    assert output.max_norm == pytest.approx(1.0)
    assert output.noise_stddev == pytest.approx(1.0)
    assert not np.array_equal(_host(backend_case, output.pytree["weight"]), 0.0)


@pytest.mark.parametrize("bound", [2.0, (-1.0, 4.0), [-0.5, 2.5]])
def test_bounded_noise_stays_inside_scalar_or_asymmetric_bound(
    backend_case, bound
) -> None:
    low, high = (-bound, bound) if isinstance(bound, float) else bound
    noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=bound, key=key(0))
    output, _ = noise_fn(
        clipped(_array(backend_case, np.zeros(10_000)), max_norm=1.0), state
    )

    observed = _host(backend_case, output.pytree)
    assert observed.min() >= low - 1e-5
    assert observed.max() <= high + 1e-5


def test_bounded_noise_clamps_zero_noise_and_preserves_unbounded_input(
    backend_case,
) -> None:
    values = _array(backend_case, [2.0, -2.0, 0.5, -0.5])
    bounded_fn, bounded_state = gaussian_noise(
        noise_multiplier=0.0, bound=1.5, key=key(0)
    )
    bounded, _ = bounded_fn(clipped(values, max_norm=1.0), bounded_state)
    identity_fn, identity_state = gaussian_noise(noise_multiplier=0.0, key=key(0))
    identity, _ = identity_fn(clipped(values, max_norm=1.0), identity_state)

    np.testing.assert_array_equal(
        _host(backend_case, bounded.pytree), np.array([1.5, -1.5, 0.5, -0.5])
    )
    np.testing.assert_array_equal(
        _host(backend_case, identity.pytree), _host(backend_case, values)
    )
    assert bounded.noise_stddev == identity.noise_stddev == 0.0


@pytest.mark.parametrize(
    ("bound", "message"),
    [
        (0.0, "scalar bound must be positive"),
        (-1.0, "scalar bound must be positive"),
        ((2.0, 1.0), "low < high"),
        ((1.0, 2.0), "straddle zero"),
        ((1.0, 2.0, 3.0), "2-tuple"),
    ],
)
def test_bounded_noise_validates_public_bound(backend_case, bound, message) -> None:
    del backend_case
    with pytest.raises(ValueError, match=message):
        gaussian_noise(noise_multiplier=1.0, bound=bound, key=key(0))


def test_bounded_noise_validates_wrappers_multiplier_and_key(backend_case) -> None:
    values = _array(backend_case, np.zeros(3))
    with pytest.raises(ValueError, match="noise_multiplier must be non-negative"):
        gaussian_noise(noise_multiplier=-1.0, bound=3.0, key=key(0))
    with pytest.raises(TypeError, match="key must be"):
        gaussian_noise(noise_multiplier=1.0, bound=3.0, key="bad")

    noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=3.0, key=key(0))
    with pytest.raises(TypeError, match="ClippedPytree"):
        noise_fn(values, state)
    with pytest.raises(TypeError, match="not NoisedPytree"):
        noise_fn(noised(values, max_norm=1.0, noise_stddev=1.0), state)
    with pytest.raises(ValueError, match="non-negative"):
        noise_fn(clipped(values, max_norm=-1.0), state)


def test_bounded_noise_handles_nested_and_tuple_pytrees(backend_case) -> None:
    nested = {
        "layer1": {
            "w": _array(backend_case, np.zeros((10, 5))),
            "b": _array(backend_case, np.zeros(10)),
        },
        "layer2": {
            "w": _array(backend_case, np.zeros((5, 3))),
            "b": _array(backend_case, np.zeros(3)),
        },
    }
    noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=5.0, key=key(0))
    nested_output, state = noise_fn(clipped(nested, max_norm=1.0), state)
    tuple_output, _ = noise_fn(
        clipped(
            (
                _array(backend_case, np.zeros((10, 5))),
                _array(backend_case, np.zeros(10)),
            ),
            max_norm=1.0,
        ),
        state,
    )

    assert set(nested_output.pytree) == {"layer1", "layer2"}
    assert not np.array_equal(
        _host(backend_case, nested_output.pytree["layer1"]["w"]), 0.0
    )
    assert len(tuple_output.pytree) == 2
    assert not np.array_equal(_host(backend_case, tuple_output.pytree[0]), 0.0)


def test_bounded_noise_is_keyed_and_advances_state(backend_case) -> None:
    gradients = clipped(_array(backend_case, np.zeros((10, 10))), max_norm=1.0)
    first_fn, first_state = gaussian_noise(noise_multiplier=1.0, bound=3.0, key=key(42))
    second_fn, second_state = gaussian_noise(
        noise_multiplier=1.0, bound=3.0, key=key(42)
    )
    other_fn, other_state = gaussian_noise(noise_multiplier=1.0, bound=3.0, key=key(43))
    first, first_state = first_fn(gradients, first_state)
    replay, _ = second_fn(gradients, second_state)
    other, _ = other_fn(gradients, other_state)
    next_output, first_state = first_fn(gradients, first_state)

    np.testing.assert_array_equal(
        _host(backend_case, first.pytree), _host(backend_case, replay.pytree)
    )
    assert not np.array_equal(
        _host(backend_case, first.pytree), _host(backend_case, other.pytree)
    )
    assert not np.array_equal(
        _host(backend_case, first.pytree), _host(backend_case, next_output.pytree)
    )
    assert first_state._step_counter == 2


def test_per_group_gaussian_noise_calibrates_and_handles_zero_groups(
    backend_case,
) -> None:
    maximum = PerGroup(
        groups={"weight": "attn", "bias": "mlp", "clean": "none"},
        values={"attn": 1.0, "mlp": 5.0, "none": 0.0},
    )
    gradients = {
        "weight": _array(backend_case, np.zeros(1_000)),
        "bias": _array(backend_case, np.zeros(1_000)),
        "clean": _array(backend_case, np.ones(1_000)),
    }
    noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=20.0, key=key(42))
    output, state = noise_fn(clipped(gradients, max_norm=maximum), state)

    assert isinstance(output.noise_stddev, PerGroup)
    assert output.noise_stddev.values == {
        "attn": pytest.approx(math.sqrt(6.0)),
        "mlp": pytest.approx(math.sqrt(30.0)),
        "none": pytest.approx(0.0),
    }
    assert (
        _host(backend_case, output.pytree["bias"]).var()
        > _host(backend_case, output.pytree["weight"]).var() * 2.5
    )
    np.testing.assert_array_equal(
        _host(backend_case, output.pytree["clean"]), np.ones(1_000)
    )
    assert state._step_counter == 1


def test_per_group_bounded_noise_resolves_nested_paths_and_validates_paths(
    backend_case,
) -> None:
    nested = {
        "layer1": {
            "attn": _array(backend_case, np.zeros(200)),
            "mlp": _array(backend_case, np.zeros(200)),
        },
        "layer2": {
            "attn": _array(backend_case, np.zeros(200)),
            "mlp": _array(backend_case, np.zeros(200)),
        },
    }
    maximum = per_group(nested, attn=1.0, mlp=5.0)
    noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=50.0, key=key(7))
    output, _ = noise_fn(clipped(nested, max_norm=maximum), state)
    assert ("layer1", "attn") in output.noise_stddev.groups
    assert ("layer2", "mlp") in output.noise_stddev.groups
    assert (
        _host(backend_case, output.pytree["layer1"]["mlp"]).var()
        > _host(backend_case, output.pytree["layer1"]["attn"]).var() * 2.5
    )

    mismatched = PerGroup(groups={"w": "g"}, values={"g": 1.0})
    with pytest.raises(KeyError):
        noise_fn(
            clipped([_array(backend_case, np.zeros(3))], max_norm=mismatched), state
        )


def test_per_group_gaussian_noise_integrates_with_clipped_gradient(
    backend_case,
) -> None:
    def loss(params, data):
        return ((params["attn_w"] * data + params["mlp_w"] * data) ** 2).mean()

    params = {"attn_w": _array(backend_case, 1.0), "mlp_w": _array(backend_case, 2.0)}
    maximum = per_group(params, attn=1.0, mlp=2.0)
    grad_fn, clip_state = clipped_grad(
        loss, argnums=0, batch_argnums=1, clipping_norm=maximum, normalize_by=10.0
    )
    noise_fn, noise_state = gaussian_noise(noise_multiplier=1.1, key=key(42))
    gradients, _ = grad_fn(
        params, _array(backend_case, np.linspace(-1.0, 1.0, 10)), state=clip_state
    )
    output, _ = noise_fn(gradients, noise_state)

    assert isinstance(output, NoisedPytree)
    assert output.noise_stddev.values == {
        "attn": pytest.approx(1.1 * math.sqrt(0.03)),
        "mlp": pytest.approx(1.1 * math.sqrt(0.06)),
    }


def test_bounded_gaussian_noise_matches_truncated_normal_distribution(
    backend_case,
) -> None:
    # gaussian_noise(bound=B) at max_norm=1 reduces to a univariate truncated
    # normal of stddev nm·1 on [-B, B] centred at zero.
    bound, sigma = 2.0, 1.0
    noise_fn, state = gaussian_noise(noise_multiplier=sigma, bound=bound, key=key(42))
    output, _ = noise_fn(
        clipped(_array(backend_case, np.zeros(50_000)), max_norm=1.0), state
    )

    _, p_value = scipy.stats.kstest(
        _host(backend_case, output.pytree),
        scipy.stats.truncnorm.cdf,
        args=(-bound / sigma, bound / sigma, 0.0, sigma),
    )
    assert p_value > 0.01, f"KS test failed with p={p_value}"


def test_bounded_gaussian_noise_variance_is_below_unclipped(backend_case) -> None:
    noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=2.0, key=key(0))
    output, _ = noise_fn(
        clipped(_array(backend_case, np.zeros(50_000)), max_norm=1.0), state
    )
    assert float(_host(backend_case, output.pytree).var()) < 1.0


def test_per_group_bounded_noise_stays_within_shared_bound(backend_case) -> None:
    # The bound is absolute and shared across groups.
    maximum = PerGroup(
        groups={"small": "lo", "large": "hi"}, values={"lo": 0.5, "hi": 2.0}
    )
    bound = 3.0
    noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=bound, key=key(0))
    gradients = {
        "small": _array(backend_case, np.zeros(10_000)),
        "large": _array(backend_case, np.zeros(10_000)),
    }
    output, _ = noise_fn(clipped(gradients, max_norm=maximum), state)

    for value in output.pytree.values():
        observed = _host(backend_case, value)
        assert observed.min() >= -bound
        assert observed.max() <= bound


def test_per_group_zero_bound_clamps_and_rejects_negative_group_bound(
    backend_case,
) -> None:
    # σ_g = 0 + absolute bound → clamp(input, low, high) per group.
    maximum = PerGroup(groups={"a": "g1", "b": "g2"}, values={"g1": 0.0, "g2": 0.0})
    bound = 0.5
    noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=bound, key=key(0))
    gradients = {
        "a": _array(backend_case, [2.0, -2.0, 0.1]),
        "b": _array(backend_case, [-1.0, 1.0, 0.3]),
    }
    output, _ = noise_fn(clipped(gradients, max_norm=maximum), state)

    np.testing.assert_allclose(
        _host(backend_case, output.pytree["a"]), np.array([bound, -bound, 0.1])
    )
    np.testing.assert_allclose(
        _host(backend_case, output.pytree["b"]), np.array([-bound, bound, 0.3])
    )

    negative = PerGroup(groups={"w": "g"}, values={"g": -1.0})
    with pytest.raises(ValueError, match="non-negative"):
        noise_fn(
            clipped({"w": _array(backend_case, np.zeros(3))}, max_norm=negative), state
        )
