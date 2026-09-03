"""Provider-neutral matrix-factorization noise behavior."""

from __future__ import annotations

import numpy as np
import pytest

from opaque import ops
from opaque.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    bsr_strategy,
    identity_strategy,
    lambda_cgd_strategy,
    mf_gaussian_noise,
)
from opaque.random import key
from opaque.serialization import from_state_dict, state_dict
from opaque.types import (
    PerGroup,
    SecondMomentClippingOutput,
    SecondMomentNoiseOutput,
    clipped,
)

_STRATEGIES = [
    ("identity", identity_strategy()),
    ("band", band_mf_strategy(bands=2, momentum=0.5)),
    ("blt", blt_strategy(max_buffers=1, momentum=0.5)),
    ("bsr", bsr_strategy(bandwidth=2, alpha=0.9, beta=0.5)),
    ("bisr", bisr_strategy(bandwidth=2, momentum=0.2)),
    ("lambda", lambda_cgd_strategy(lambda_=0.5)),
]


def _template(backend_case, *, dtype: str = "float32"):
    backend_dtype = backend_case.dtype(dtype)
    return {
        "weight": backend_case.array(np.zeros((3, 2)), dtype=backend_dtype),
        "bias": backend_case.array(np.zeros((2,)), dtype=backend_dtype),
    }


def _noise(backend_case, strategy, *, seed: int = 7, n_steps: int = 4, **kwargs):
    return mf_gaussian_noise(
        _template(backend_case),
        strategy,
        n_steps=n_steps,
        noise_multiplier=0.5,
        key=key(seed),
        **kwargs,
    )


def _vector_template(backend_case, *, size: int = 10, fill: float = 0.0):
    return {
        "w": backend_case.array(
            np.full((size,), fill, dtype=np.float32),
            dtype=backend_case.dtype("float32"),
        )
    }


def _lambda_noise(
    backend_case,
    *,
    template=None,
    n_steps: int = 100,
    lambda_: float = 0.9,
    normalized: bool = True,
    seed: int = 42,
):
    return mf_gaussian_noise(
        _vector_template(backend_case) if template is None else template,
        lambda_cgd_strategy(lambda_=lambda_, normalized=normalized),
        n_steps=n_steps,
        min_sep=1,
        max_participations=1,
        noise_multiplier=1.0,
        key=key(seed),
    )


def _grads(backend_case, *, maximum: float | PerGroup = 1.0):
    return clipped(_template(backend_case), max_norm=maximum)


@pytest.mark.parametrize(("name", "strategy"), _STRATEGIES)
def test_mf_noise_preserves_shape_dtype_and_state(
    backend_case, name: str, strategy
) -> None:
    del name
    noise_fn, state = _noise(backend_case, strategy)
    first, state = noise_fn(_grads(backend_case), state)
    second, state = noise_fn(_grads(backend_case), state)

    assert first.pytree["weight"].shape == (3, 2)
    assert first.pytree["weight"].dtype == backend_case.dtype("float32")
    assert first.noise_stddev > 0.0
    assert state._step_counter == 2
    assert not np.array_equal(
        backend_case.to_host(first.pytree["weight"]),
        backend_case.to_host(second.pytree["weight"]),
    )


@pytest.mark.parametrize(("name", "strategy"), _STRATEGIES)
def test_mf_noise_rejects_calls_past_calibrated_horizon(
    backend_case, name: str, strategy
) -> None:
    del name
    noise_fn, state = _noise(backend_case, strategy, n_steps=2)
    gradients = _grads(backend_case)
    _, state = noise_fn(gradients, state)
    _, state = noise_fn(gradients, state)
    with pytest.raises(ValueError, match="outside the calibrated horizon"):
        noise_fn(gradients, state)


def test_mf_noise_checkpoint_replays_next_streaming_step(backend_case) -> None:
    strategy = band_mf_strategy(bands=2, momentum=0.8)
    noise_fn, state = _noise(backend_case, strategy, seed=31)
    gradients = _grads(backend_case)
    _, state = noise_fn(gradients, state)
    snapshot = state_dict(state)
    uninterrupted, uninterrupted_state = noise_fn(gradients, state)
    _, template = _noise(backend_case, strategy, seed=999)
    restored = from_state_dict(template, snapshot)
    resumed, resumed_state = noise_fn(gradients, restored)

    np.testing.assert_array_equal(
        backend_case.to_host(uninterrupted.pytree["weight"]),
        backend_case.to_host(resumed.pytree["weight"]),
    )
    assert uninterrupted_state._step_counter == resumed_state._step_counter == 2


def test_band_mf_state_dict_continues_with_saved_streaming_state(backend_case) -> None:
    strategy = band_mf_strategy(bands=2, momentum=0.8)
    template = _vector_template(backend_case, size=4)
    noise_fn, state = mf_gaussian_noise(
        template,
        strategy,
        n_steps=4,
        min_sep=1,
        max_participations=4,
        noise_multiplier=1.0,
        key=key(42),
    )
    gradients = clipped(template, max_norm=1.0)

    _, state = noise_fn(gradients, state)
    _, state = noise_fn(gradients, state)
    snapshot = state_dict(state)

    _, restore_template = mf_gaussian_noise(
        template,
        strategy,
        n_steps=4,
        min_sep=1,
        max_participations=4,
        noise_multiplier=1.0,
        key=key(99),
    )
    restored = from_state_dict(restore_template, snapshot)

    expected, expected_state = noise_fn(gradients, state)
    actual, actual_state = noise_fn(gradients, restored)
    np.testing.assert_allclose(
        backend_case.to_host(actual.pytree["w"]),
        backend_case.to_host(expected.pytree["w"]),
    )
    assert actual_state._step_counter == expected_state._step_counter


def test_mf_noise_calibrates_per_group_and_paired_releases(backend_case) -> None:
    maximum = PerGroup(
        groups={"weight": "weight", "bias": "bias"},
        values={"weight": 0.5, "bias": 1.0},
    )
    noise_fn, state = _noise(
        backend_case,
        identity_strategy(),
        second_moment_strategy=identity_strategy(),
    )
    gradients = _template(backend_case)
    paired = SecondMomentClippingOutput(
        grads=clipped(gradients, max_norm=maximum),
        squared_grads=clipped(
            {name: ops.multiply(value, value) for name, value in gradients.items()},
            max_norm=maximum * maximum,
        ),
    )
    output, state = noise_fn(paired, state)

    assert isinstance(output, SecondMomentNoiseOutput)
    assert isinstance(output.noisy_grads.noise_stddev, PerGroup)
    assert isinstance(output.noisy_squared_grads.noise_stddev, PerGroup)
    assert (
        output.noisy_grads.noise_stddev.values["bias"]
        > output.noisy_grads.noise_stddev.values["weight"]
    )
    assert state._step_counter == 1
    assert not np.array_equal(
        backend_case.to_host(output.noisy_grads.pytree["weight"]),
        backend_case.to_host(output.noisy_squared_grads.pytree["weight"]),
    )


def test_identity_mf_noise_reduces_to_public_noise_scale(backend_case) -> None:
    noise_fn, state = _noise(backend_case, identity_strategy())
    output, _ = noise_fn(_grads(backend_case, maximum=2.0), state)
    assert output.noise_stddev == pytest.approx(1.0)


def test_identity_mf_noise_preserves_pytree_shape_and_scale(backend_case) -> None:
    template = _template(backend_case)
    large_fn, large_state = mf_gaussian_noise(
        template,
        identity_strategy(),
        n_steps=3,
        noise_multiplier=100.0,
        key=key(101),
    )
    small_fn, small_state = mf_gaussian_noise(
        template,
        identity_strategy(),
        n_steps=3,
        noise_multiplier=1.0,
        key=key(103),
    )
    large, large_state = large_fn(_grads(backend_case), large_state)
    small, small_state = small_fn(_grads(backend_case), small_state)

    assert set(large.pytree) == {"weight", "bias"}
    assert large.pytree["weight"].shape == (3, 2)
    assert large_state._step_counter == small_state._step_counter == 1
    assert (
        backend_case.to_host(large.pytree["weight"]).std()
        > backend_case.to_host(small.pytree["weight"]).std() * 10
    )


def test_lambda_cgd_replays_its_keyed_correlation(backend_case) -> None:
    zeros = _template(backend_case)
    correlated_fn, correlated_state = _noise(
        backend_case, lambda_cgd_strategy(lambda_=0.5, normalized=False), seed=43
    )
    iid_fn, iid_state = _noise(
        backend_case, lambda_cgd_strategy(lambda_=0.0, normalized=False), seed=43
    )
    correlated_first, correlated_state = correlated_fn(
        clipped(zeros, max_norm=1.0), correlated_state
    )
    correlated_second, _ = correlated_fn(clipped(zeros, max_norm=1.0), correlated_state)
    iid_first, iid_state = iid_fn(clipped(zeros, max_norm=1.0), iid_state)
    iid_second, _ = iid_fn(clipped(zeros, max_norm=1.0), iid_state)

    np.testing.assert_allclose(
        backend_case.to_host(correlated_first.pytree["weight"]),
        backend_case.to_host(iid_first.pytree["weight"]),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        backend_case.to_host(correlated_second.pytree["weight"]),
        backend_case.to_host(iid_second.pytree["weight"])
        - 0.5 * backend_case.to_host(iid_first.pytree["weight"]),
        rtol=1e-5,
        atol=1e-6,
    )


def test_lambda_cgd_basic_noise_generation(backend_case) -> None:
    template = _vector_template(backend_case)
    noise_fn, state = _lambda_noise(backend_case, template=template)
    output, new_state = noise_fn(clipped(template, max_norm=1.0), state)

    assert output.pytree["w"].shape == (10,)
    assert new_state._step_counter == 1


def test_lambda_cgd_is_deterministic_with_same_key(backend_case) -> None:
    results = []
    template = _vector_template(backend_case)
    for _ in range(2):
        noise_fn, state = _lambda_noise(backend_case, template=template)
        first, state = noise_fn(clipped(template, max_norm=1.0), state)
        second, state = noise_fn(clipped(template, max_norm=1.0), state)
        results.append(
            np.concatenate(
                [
                    backend_case.to_host(first.pytree["w"]),
                    backend_case.to_host(second.pytree["w"]),
                ]
            )
        )

    np.testing.assert_allclose(results[0], results[1])


def test_lambda_cgd_different_keys_give_different_noise(backend_case) -> None:
    template = _vector_template(backend_case)
    first_fn, first_state = _lambda_noise(backend_case, template=template, seed=1)
    second_fn, second_state = _lambda_noise(backend_case, template=template, seed=2)
    first, _ = first_fn(clipped(template, max_norm=1.0), first_state)
    second, _ = second_fn(clipped(template, max_norm=1.0), second_state)

    assert not np.allclose(
        backend_case.to_host(first.pytree["w"]),
        backend_case.to_host(second.pytree["w"]),
    )


def test_lambda_cgd_zero_is_independent(backend_case) -> None:
    template = _vector_template(backend_case)
    noise_fn, state = _lambda_noise(backend_case, template=template, lambda_=0.0)
    first, state = noise_fn(clipped(template, max_norm=1.0), state)
    second, _ = noise_fn(clipped(template, max_norm=1.0), state)

    assert backend_case.to_host(first.pytree["w"]).std() > 0.1
    assert backend_case.to_host(second.pytree["w"]).std() > 0.1


def test_lambda_cgd_handles_multi_parameter_templates(backend_case) -> None:
    backend_dtype = backend_case.dtype("float32")
    template = {
        "w1": backend_case.array(np.zeros(5, dtype=np.float32), dtype=backend_dtype),
        "w2": backend_case.array(
            np.zeros((3, 4), dtype=np.float32), dtype=backend_dtype
        ),
    }
    noise_fn, state = _lambda_noise(backend_case, template=template)
    output, _ = noise_fn(clipped(template, max_norm=1.0), state)

    assert output.pytree["w1"].shape == (5,)
    assert output.pytree["w2"].shape == (3, 4)


def test_lambda_cgd_noise_adds_to_gradients(backend_case) -> None:
    template = _vector_template(backend_case)
    noise_fn, state = _lambda_noise(backend_case, template=template)
    noised, _ = noise_fn(clipped(_vector_template(backend_case, fill=5.0), max_norm=1.0), state)
    noise_only_fn, noise_only_state = _lambda_noise(backend_case, template=template)
    noise_only, _ = noise_only_fn(clipped(template, max_norm=1.0), noise_only_state)

    np.testing.assert_allclose(
        backend_case.to_host(noised.pytree["w"]) - 5.0,
        backend_case.to_host(noise_only.pytree["w"]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_lambda_cgd_step_counter_increments(backend_case) -> None:
    template = _vector_template(backend_case)
    noise_fn, state = _lambda_noise(backend_case, template=template)
    assert state._step_counter == 0

    _, state = noise_fn(clipped(template, max_norm=1.0), state)
    assert state._step_counter == 1
    _, state = noise_fn(clipped(template, max_norm=1.0), state)
    assert state._step_counter == 2


def test_mf_noise_compute_dtype_keeps_public_output_dtype(backend_case) -> None:
    template = _template(backend_case)
    noise_fn, state = mf_gaussian_noise(
        template,
        identity_strategy(),
        n_steps=2,
        noise_multiplier=1.0,
        compute_dtype=backend_case.dtype("float32"),
        key=key(59),
    )
    output, _ = noise_fn(clipped(template, max_norm=1.0), state)
    assert output.pytree["weight"].dtype == backend_case.dtype("float32")


def test_mf_noise_default_compute_dtype_matches_explicit_float32(backend_case) -> None:
    template = _template(backend_case)
    gradients = clipped(template, max_norm=1.0)
    default_fn, default_state = mf_gaussian_noise(
        template,
        identity_strategy(),
        n_steps=2,
        noise_multiplier=1.0,
        key=key(109),
    )
    explicit_fn, explicit_state = mf_gaussian_noise(
        template,
        identity_strategy(),
        n_steps=2,
        noise_multiplier=1.0,
        compute_dtype=backend_case.dtype("float32"),
        key=key(109),
    )
    default, _ = default_fn(gradients, default_state)
    explicit, _ = explicit_fn(gradients, explicit_state)

    np.testing.assert_array_equal(
        backend_case.to_host(default.pytree["weight"]),
        backend_case.to_host(explicit.pytree["weight"]),
    )


@pytest.mark.parametrize("strategy", [identity_strategy(), band_mf_strategy(bands=2)])
def test_mf_noise_requires_an_integer_horizon(backend_case, strategy) -> None:
    with pytest.raises(TypeError, match="n_steps must be an int"):
        mf_gaussian_noise(
            _template(backend_case),
            strategy,
            n_steps=2.5,
            noise_multiplier=1.0,
            key=key(61),
        )


def test_public_identity_matches_identity_execution_plan(backend_case):
    import numpy as np

    from opaque.api.dpftrl.noise._engine import _matrix_factorization_noise
    from opaque.api.dpftrl.noise._plan import identity_execution_plan

    template = {"w": backend_case.array(np.zeros(10, dtype=np.float32))}
    public_fn, public_state = mf_gaussian_noise(
        template,
        identity_strategy(),
        n_steps=10,
        noise_multiplier=1.0,
        key=key(42),
    )
    plan_fn, plan_state = _matrix_factorization_noise(
        template, identity_execution_plan(10), key=key(42)
    )
    grad = {"w": backend_case.array(np.ones(10, dtype=np.float32))}
    public, _ = public_fn(clipped(grad, max_norm=1.0), public_state)
    planned, _ = plan_fn(grad, plan_state, stddev=1.0)
    backend_case.assert_allclose(public.pytree["w"], planned["w"])
