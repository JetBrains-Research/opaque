"""Provider-neutral private second-moment MF noise behavior."""

from __future__ import annotations

import numpy as np
import pytest

from opaque.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    bsr_strategy,
    identity_strategy,
    lambda_cgd_strategy,
    mf_gaussian_noise,
)
from opaque.dpftrl.noise.types import SecondMomentMFNoiseState
from opaque.random import key
from opaque.types import (
    NoisedPytree,
    PerGroup,
    SecondMomentClippingOutput,
    SecondMomentNoiseOutput,
    clipped,
)

_SENSITIVITY = 0.1
_N_STEPS = 50


def _zeros(
    backend_case,
    *,
    w_shape: tuple[int, ...] = (4, 3),
    b_shape: tuple[int, ...] = (4,),
) -> dict[str, object]:
    backend_dtype = backend_case.dtype("float32")
    return {
        "w": backend_case.array(np.zeros(w_shape, dtype=np.float32), dtype=backend_dtype),
        "b": backend_case.array(np.zeros(b_shape, dtype=np.float32), dtype=backend_dtype),
    }


def _ones(
    backend_case,
    *,
    w_shape: tuple[int, ...] = (4, 3),
    b_shape: tuple[int, ...] = (4,),
) -> dict[str, object]:
    backend_dtype = backend_case.dtype("float32")
    return {
        "w": backend_case.array(np.ones(w_shape, dtype=np.float32), dtype=backend_dtype),
        "b": backend_case.array(np.ones(b_shape, dtype=np.float32), dtype=backend_dtype),
    }


def _random(
    backend_case,
    *,
    seed: int,
    w_shape: tuple[int, ...] = (4, 3),
    b_shape: tuple[int, ...] = (4,),
) -> dict[str, object]:
    backend_dtype = backend_case.dtype("float32")
    rng = np.random.RandomState(seed)
    return {
        "w": backend_case.array(rng.randn(*w_shape).astype(np.float32), dtype=backend_dtype),
        "b": backend_case.array(rng.randn(*b_shape).astype(np.float32), dtype=backend_dtype),
    }


def _make_noise(
    backend_case,
    strategy,
    second_strategy,
    *,
    n_steps: int = _N_STEPS,
    min_sep: int = _N_STEPS,
    max_participations: int | None = 1,
    noise_multiplier: float = 1.0,
    seed: int = 42,
    template: dict[str, object] | None = None,
):
    return mf_gaussian_noise(
        _zeros(backend_case) if template is None else template,
        strategy,
        n_steps=n_steps,
        min_sep=min_sep,
        max_participations=max_participations,
        noise_multiplier=noise_multiplier,
        key=key(seed),
        second_moment_strategy=second_strategy,
    )


def _second_moment_input(
    grads: dict[str, object], *, max_norm: float | PerGroup = _SENSITIVITY
) -> SecondMomentClippingOutput:
    squared = {name: value * value for name, value in grads.items()}
    return SecondMomentClippingOutput(
        grads=clipped(grads, max_norm=max_norm),
        squared_grads=clipped(squared, max_norm=max_norm * max_norm),
    )


def _max_column_norm(strategy, *, n_steps: int) -> float:
    return strategy.sensitivity(n_steps=n_steps, min_sep=n_steps, max_participations=1)


def _row_l2_at_zero(strategy, *, n_steps: int, min_sep: int = 1) -> float:
    plan = strategy.execution_plan(
        n_steps=n_steps, min_sep=min_sep, max_participations=1
    )
    return plan.row_l2[0]


@pytest.mark.parametrize("noise_multiplier", [0.5, 1.0, 2.0])
def test_second_moment_joint_mahalanobis_matches_mf_gaussian_pld(
    backend_case, noise_multiplier: float
) -> None:
    strategy = blt_strategy(momentum=0.9)
    second_strategy = blt_strategy(momentum=0.99)
    c1 = _max_column_norm(strategy, n_steps=_N_STEPS)
    c2 = _max_column_norm(second_strategy, n_steps=_N_STEPS)

    noise_fn, state = _make_noise(
        backend_case,
        strategy,
        second_strategy,
        noise_multiplier=noise_multiplier,
    )
    output, _ = noise_fn(_second_moment_input(_zeros(backend_case)), state)

    first_row_l2 = _row_l2_at_zero(strategy, n_steps=_N_STEPS, min_sep=_N_STEPS)
    second_row_l2 = _row_l2_at_zero(
        second_strategy, n_steps=_N_STEPS, min_sep=_N_STEPS
    )
    base_sigma_first = output.noisy_grads.noise_stddev / first_row_l2
    base_sigma_second = output.noisy_squared_grads.noise_stddev / second_row_l2
    mahalanobis = (
        (_SENSITIVITY * c1 / base_sigma_first) ** 2
        + ((_SENSITIVITY**2) * c2 / base_sigma_second) ** 2
    )

    assert mahalanobis == pytest.approx((c1 / noise_multiplier) ** 2, rel=1e-10)


def test_second_moment_first_stream_recovers_single_stream_in_small_squared_limit(
    backend_case,
) -> None:
    strategy = blt_strategy(momentum=0.9)
    second_strategy = blt_strategy(momentum=0.99)
    small_sensitivity = 1e-6
    noise_fn, state = _make_noise(backend_case, strategy, second_strategy)
    paired = _second_moment_input(_zeros(backend_case), max_norm=small_sensitivity)

    output, _ = noise_fn(paired, state)
    first_row_l2 = _row_l2_at_zero(strategy, n_steps=_N_STEPS, min_sep=_N_STEPS)
    base_sigma_first = output.noisy_grads.noise_stddev / first_row_l2
    assert base_sigma_first == pytest.approx(small_sensitivity, rel=1e-3)


def test_second_moment_joint_budget_with_per_group_bounds(backend_case) -> None:
    strategy = band_mf_strategy(bands=4, momentum=0.9)
    second_strategy = band_mf_strategy(bands=4, momentum=0.99)
    noise_multiplier = 1.1
    c1 = _max_column_norm(strategy, n_steps=_N_STEPS)
    c2 = _max_column_norm(second_strategy, n_steps=_N_STEPS)
    per_group = PerGroup(
        groups={"w": "a", "b": "b"},
        values={"a": 0.04, "b": 0.08},
    )
    template = _zeros(backend_case, w_shape=(3, 2), b_shape=(3,))
    paired = _second_moment_input(_ones(backend_case, w_shape=(3, 2), b_shape=(3,)), max_norm=per_group)

    noise_fn, state = _make_noise(
        backend_case,
        strategy,
        second_strategy,
        template=template,
        noise_multiplier=noise_multiplier,
    )
    output, _ = noise_fn(paired, state)

    first_stddev = output.noisy_grads.noise_stddev
    second_stddev = output.noisy_squared_grads.noise_stddev
    assert isinstance(first_stddev, PerGroup)
    assert isinstance(second_stddev, PerGroup)

    first_row_l2 = _row_l2_at_zero(strategy, n_steps=_N_STEPS, min_sep=_N_STEPS)
    second_row_l2 = _row_l2_at_zero(
        second_strategy, n_steps=_N_STEPS, min_sep=_N_STEPS
    )
    mahalanobis = 0.0
    squared_group = per_group * per_group
    for param_name in ("w", "b"):
        delta_first = per_group.for_path(param_name) * c1
        delta_second = squared_group.for_path(param_name) * c2
        base_first = first_stddev.for_path(param_name) / first_row_l2
        base_second = second_stddev.for_path(param_name) / second_row_l2
        mahalanobis += (delta_first / base_first) ** 2 + (delta_second / base_second) ** 2

    assert mahalanobis == pytest.approx((c1 / noise_multiplier) ** 2, rel=1e-9)


class TestSecondMomentMFNoise:
    @pytest.fixture
    def grad_template(self, backend_case):
        return _zeros(backend_case)

    @pytest.fixture
    def grads(self, backend_case):
        return _random(backend_case, seed=7)

    def test_returns_correct_types(self, backend_case, grad_template, grads) -> None:
        noise_fn, state = _make_noise(
            backend_case,
            band_mf_strategy(bands=5, momentum=0.9),
            band_mf_strategy(bands=5, momentum=0.99),
            template=grad_template,
        )
        assert isinstance(state, SecondMomentMFNoiseState)

        output, new_state = noise_fn(_second_moment_input(grads), state)
        assert isinstance(output, SecondMomentNoiseOutput)
        assert isinstance(output.noisy_grads, NoisedPytree)
        assert isinstance(output.noisy_squared_grads, NoisedPytree)
        assert isinstance(new_state, SecondMomentMFNoiseState)

    def test_output_shapes_match_input(self, backend_case, grad_template, grads) -> None:
        noise_fn, state = _make_noise(
            backend_case,
            band_mf_strategy(bands=5, momentum=0.9),
            band_mf_strategy(bands=5, momentum=0.99),
            template=grad_template,
        )
        output, _ = noise_fn(_second_moment_input(grads), state)

        assert output.noisy_grads.pytree["w"].shape == (4, 3)
        assert output.noisy_grads.pytree["b"].shape == (4,)
        assert output.noisy_squared_grads.pytree["w"].shape == (4, 3)
        assert output.noisy_squared_grads.pytree["b"].shape == (4,)

    def test_tuple_unpacking(self, backend_case, grad_template, grads) -> None:
        noise_fn, state = _make_noise(
            backend_case,
            band_mf_strategy(bands=5, momentum=0.9),
            band_mf_strategy(bands=5, momentum=0.99),
            template=grad_template,
        )

        noisy_grads, noisy_squared = noise_fn(_second_moment_input(grads), state)[0]
        assert isinstance(noisy_grads, NoisedPytree)
        assert isinstance(noisy_squared, NoisedPytree)

    def test_step_counter_increments(self, backend_case, grad_template, grads) -> None:
        noise_fn, state = _make_noise(
            backend_case,
            band_mf_strategy(bands=5, momentum=0.9),
            band_mf_strategy(bands=5, momentum=0.99),
            template=grad_template,
        )

        assert state._step_counter == 0
        _, state = noise_fn(_second_moment_input(grads), state)
        assert state._step_counter == 1
        _, state = noise_fn(_second_moment_input(grads), state)
        assert state._step_counter == 2

    def test_deterministic_with_same_key(self, backend_case, grad_template, grads) -> None:
        noise_fn1, state1 = _make_noise(
            backend_case,
            band_mf_strategy(bands=5, momentum=0.9),
            band_mf_strategy(bands=5, momentum=0.99),
            template=grad_template,
        )
        noise_fn2, state2 = _make_noise(
            backend_case,
            band_mf_strategy(bands=5, momentum=0.9),
            band_mf_strategy(bands=5, momentum=0.99),
            template=grad_template,
        )

        output1, _ = noise_fn1(_second_moment_input(grads), state1)
        output2, _ = noise_fn2(_second_moment_input(grads), state2)
        np.testing.assert_allclose(
            backend_case.to_host(output1.noisy_grads.pytree["w"]),
            backend_case.to_host(output2.noisy_grads.pytree["w"]),
        )
        np.testing.assert_allclose(
            backend_case.to_host(output1.noisy_squared_grads.pytree["w"]),
            backend_case.to_host(output2.noisy_squared_grads.pytree["w"]),
        )

    def test_different_keys_give_different_noise(
        self, backend_case, grad_template, grads
    ) -> None:
        noise_fn1, state1 = _make_noise(
            backend_case,
            band_mf_strategy(bands=5, momentum=0.9),
            band_mf_strategy(bands=5, momentum=0.99),
            template=grad_template,
            seed=42,
        )
        noise_fn2, state2 = _make_noise(
            backend_case,
            band_mf_strategy(bands=5, momentum=0.9),
            band_mf_strategy(bands=5, momentum=0.99),
            template=grad_template,
            seed=99,
        )

        output1, _ = noise_fn1(_second_moment_input(grads), state1)
        output2, _ = noise_fn2(_second_moment_input(grads), state2)
        assert not np.allclose(
            backend_case.to_host(output1.noisy_grads.pytree["w"]),
            backend_case.to_host(output2.noisy_grads.pytree["w"]),
        )

    @pytest.mark.parametrize(
        ("strategy", "second_strategy"),
        [
            (
                band_mf_strategy(bands=5, momentum=0.9),
                band_mf_strategy(bands=5, momentum=0.99),
            ),
            (blt_strategy(momentum=0.9), blt_strategy(momentum=0.99)),
            (
                bisr_strategy(bandwidth=4, momentum=0.9),
                bisr_strategy(bandwidth=4, momentum=0.99),
            ),
            (
                bsr_strategy(bandwidth=4, alpha=1.0, beta=0.9),
                bsr_strategy(bandwidth=4, alpha=1.0, beta=0.99),
            ),
            (identity_strategy(), identity_strategy()),
        ],
        ids=["band_mf", "blt", "bisr", "bsr", "identity"],
    )
    def test_works_with_supported_mechanisms(
        self, backend_case, grad_template, grads, strategy, second_strategy
    ) -> None:
        noise_fn, state = _make_noise(
            backend_case,
            strategy,
            second_strategy,
            template=grad_template,
        )

        output, new_state = noise_fn(_second_moment_input(grads), state)
        assert output.noisy_grads.pytree["w"].shape == (4, 3)
        assert output.noisy_squared_grads.pytree["w"].shape == (4, 3)
        assert isinstance(new_state, SecondMomentMFNoiseState)

    def test_paired_input_requires_second_moment_strategy(
        self, backend_case, grad_template, grads
    ) -> None:
        noise_fn, state = mf_gaussian_noise(
            grad_template,
            lambda_cgd_strategy(lambda_=0.9),
            n_steps=_N_STEPS,
            min_sep=_N_STEPS,
            max_participations=1,
            noise_multiplier=1.0,
            key=key(42),
        )
        with pytest.raises(TypeError, match="second_moment_strategy"):
            noise_fn(_second_moment_input(grads), state)

    def test_single_input_rejected_when_second_moment_strategy_supplied(
        self, backend_case, grad_template, grads
    ) -> None:
        noise_fn, state = _make_noise(
            backend_case,
            lambda_cgd_strategy(lambda_=0.9),
            lambda_cgd_strategy(lambda_=0.999),
            template=grad_template,
        )
        with pytest.raises(TypeError, match="SecondMomentClippingOutput"):
            noise_fn(clipped(grads, max_norm=_SENSITIVITY), state)

    def test_lambda_cgd_accepts_explicit_second_strategy(
        self, backend_case, grad_template, grads
    ) -> None:
        noise_fn, state = _make_noise(
            backend_case,
            lambda_cgd_strategy(lambda_=0.9),
            lambda_cgd_strategy(lambda_=0.999),
            template=grad_template,
        )

        output, _ = noise_fn(_second_moment_input(grads), state)
        assert output.noisy_squared_grads.pytree["w"].shape == (4, 3)

    def test_squared_grads_are_noised_not_raw(
        self, backend_case, grad_template
    ) -> None:
        grads = _ones(backend_case)
        noise_fn, state = _make_noise(
            backend_case,
            band_mf_strategy(bands=5, momentum=0.9),
            band_mf_strategy(bands=5, momentum=0.99),
            template=grad_template,
        )

        output, _ = noise_fn(_second_moment_input(grads), state)
        assert not np.allclose(
            backend_case.to_host(output.noisy_squared_grads.pytree["w"]),
            np.ones((4, 3), dtype=np.float32),
            atol=1e-6,
        )


def test_paired_state_has_a_registered_sync_handler(backend_case) -> None:
    from opaque.distributed import sync

    _, state = _make_noise(
        backend_case,
        identity_strategy(),
        identity_strategy(),
        n_steps=2,
        min_sep=1,
        max_participations=None,
        template=_zeros(backend_case),
    )
    assert isinstance(state, SecondMomentMFNoiseState)
    assert sync(state) is state
