"""Tests for private second-moment MF noise."""

import math

import pytest
import torch

from opaque.clipping.types import ClippedPytree, clipped
from opaque.core.noise import (
    NoisedPytree,
    SecondMomentClippingOutput,
    SecondMomentNoiseOutput,
    second_moment_joint_sensitivity,
    second_moment_noise_scale,
    second_moment_stddevs,
)
from opaque.dpftrl.noise import (
    SecondMomentMFNoiseState,
    band_mf_strategy,
    bisr_strategy,
    bsr_strategy,
    blt_strategy,
    identity_strategy,
    lambda_cgd_strategy,
    mf_noise,
)
from opaque.random import key


_SENSITIVITY = 0.1


def _paired(grads):
    """Build a SecondMomentClippingOutput directly from raw grads.

    Tests pass already-computed gradients (no clipping loop), so we
    construct the paired form by hand: each stream gets its own
    ``ClippedPytree`` with the appropriate max_norm bound.  The squared
    stream's payload is element-wise g² and its bound is the squared
    contribution bound.
    """
    import torch
    grads_clipped = clipped(grads, max_norm=_SENSITIVITY)
    sq_pytree = {k: v * v for k, v in grads.items()}
    sq_clipped = clipped(sq_pytree, max_norm=_SENSITIVITY * _SENSITIVITY)
    return SecondMomentClippingOutput(
        grads=grads_clipped, squared_grads=sq_clipped
    )


def _clipped(grads):
    """Wrap raw grad pytree as ClippedPytree at the test's standard max_norm."""
    return clipped(grads, max_norm=_SENSITIVITY)


class TestSecondMomentCalibration:
    def test_joint_sensitivity_default_overhead(self):
        sensitivity = second_moment_joint_sensitivity(1.5, 0.1)
        expected = 0.1 * 1.5 * math.sqrt(1.5)
        assert sensitivity == pytest.approx(expected, rel=1e-10)

    def test_noise_scale_default_overhead(self):
        # second_moment_noise_scale now takes Δ_first and Δ_second separately.
        # For Δ_first=0.5 and Δ_second=0.25 (= Δ_first² — old "square the
        # average" math), c1=2, c2=1, scale = 0.25 / (2 · 0.5 · √0.5).
        scale = second_moment_noise_scale(
            c1_max_column_norm=2.0,
            c2_max_column_norm=1.0,
            first_max_norm=0.5,
            squared_max_norm=0.25,
        )
        expected = 0.25 / (2.0 * 0.5 * math.sqrt(0.5))
        assert scale == pytest.approx(expected, rel=1e-10)

    def test_stddevs(self):
        # Per-example correct: first_max_norm=0.2, squared_max_norm=0.04.
        # (0.04 = 0.2² coincidentally matches the old "square the average"
        # math because here Δ_first² = (0.2)² = 0.04 and we set squared
        # explicitly to that value.)
        first, second = second_moment_stddevs(
            3.0,
            first_max_norm=0.2,
            squared_max_norm=0.04,
            c1_max_column_norm=2.0,
            c2_max_column_norm=1.5,
        )
        expected_first = 3.0 * 0.2 * 2.0 * math.sqrt(1.5)
        expected_second = expected_first * (
            0.04 * 1.5 / (0.2 * 2.0 * math.sqrt(0.5))
        )
        assert first == pytest.approx(expected_first, rel=1e-10)
        assert second == pytest.approx(expected_second, rel=1e-10)

    def test_custom_overhead(self):
        first, second = second_moment_stddevs(
            1.0,
            first_max_norm=0.5,
            squared_max_norm=0.25,
            c1_max_column_norm=2.0,
            c2_max_column_norm=1.0,
            first_moment_overhead=1.25,
        )
        assert first == pytest.approx(1.25)
        assert second == pytest.approx(
            1.25 * 0.25 / (0.5 * 2.0 * math.sqrt(1.25**2 - 1.0))
        )

    def test_per_example_squared_takes_more_noise(self):
        """Per-example squared (Δ_y = C²/n) gets a factor n more noise on
        the second stream than the old (Δ_y = (C/n)²) math, when n > 1."""
        n = 16
        C = 1.0
        # Per-example correct
        _, σ2_per_example = second_moment_stddevs(
            1.0, first_max_norm=C / n, squared_max_norm=C**2 / n
        )
        # Old (wrong) math: Δ_y = (C/n)² = C²/n²
        _, σ2_old = second_moment_stddevs(
            1.0, first_max_norm=C / n, squared_max_norm=(C / n) ** 2
        )
        # Per-example correct gives n× more noise on the second stream.
        assert σ2_per_example == pytest.approx(σ2_old * n, rel=1e-10)

    def test_rejects_invalid(self):
        with pytest.raises(ValueError):
            second_moment_joint_sensitivity(0.0, 1.0)
        with pytest.raises(ValueError):
            second_moment_noise_scale(
                c1_max_column_norm=1.0,
                c2_max_column_norm=1.0,
                first_max_norm=0.0,
                squared_max_norm=1.0,
            )
        with pytest.raises(ValueError):
            second_moment_stddevs(-1.0, first_max_norm=1.0, squared_max_norm=1.0)


class TestSecondMomentMFNoise:
    @pytest.fixture
    def grad_template(self):
        return {"w": torch.zeros(4, 3), "b": torch.zeros(4)}

    def test_returns_correct_types(self, grad_template):
        strategy = band_mf_strategy(n_steps=50, bands=5, momentum=0.9)
        second_strategy = band_mf_strategy(n_steps=50, bands=5, momentum=0.99)
        noise_fn, state = mf_noise(
            grad_template,
            strategy,
            noise_multiplier=1.0,
            key=key(42),
            second_moment_strategy=second_strategy,
        )
        assert isinstance(state, SecondMomentMFNoiseState)

        grads = {"w": torch.randn(4, 3), "b": torch.randn(4)}
        output, new_state = noise_fn(_paired(grads), state)
        assert isinstance(output, SecondMomentNoiseOutput)
        assert isinstance(output.noisy_grads, NoisedPytree)
        assert isinstance(output.noisy_squared_grads, NoisedPytree)
        assert isinstance(new_state, SecondMomentMFNoiseState)

    def test_output_shapes_match_input(self, grad_template):
        strategy = band_mf_strategy(n_steps=50, bands=5, momentum=0.9)
        second_strategy = band_mf_strategy(n_steps=50, bands=5, momentum=0.99)
        noise_fn, state = mf_noise(
            grad_template,
            strategy,
            noise_multiplier=1.0,
            key=key(42),
            second_moment_strategy=second_strategy,
        )
        grads = {"w": torch.randn(4, 3), "b": torch.randn(4)}
        output, state = noise_fn(_paired(grads), state)
        assert output.noisy_grads.pytree["w"].shape == (4, 3)
        assert output.noisy_grads.pytree["b"].shape == (4,)
        assert output.noisy_squared_grads.pytree["w"].shape == (4, 3)
        assert output.noisy_squared_grads.pytree["b"].shape == (4,)

    def test_tuple_unpacking(self, grad_template):
        strategy = band_mf_strategy(n_steps=50, bands=5, momentum=0.9)
        second_strategy = band_mf_strategy(n_steps=50, bands=5, momentum=0.99)
        noise_fn, state = mf_noise(
            grad_template,
            strategy,
            noise_multiplier=1.0,
            key=key(42),
            second_moment_strategy=second_strategy,
        )
        grads = {"w": torch.randn(4, 3), "b": torch.randn(4)}
        noisy_g, noisy_sq = noise_fn(_paired(grads), state)[0]
        assert isinstance(noisy_g, NoisedPytree)
        assert isinstance(noisy_sq, NoisedPytree)

    def test_step_counter_increments(self, grad_template):
        strategy = band_mf_strategy(n_steps=50, bands=5, momentum=0.9)
        second_strategy = band_mf_strategy(n_steps=50, bands=5, momentum=0.99)
        noise_fn, state = mf_noise(
            grad_template,
            strategy,
            noise_multiplier=1.0,
            key=key(42),
            second_moment_strategy=second_strategy,
        )
        assert state._step_counter == 0
        grads = {"w": torch.randn(4, 3), "b": torch.randn(4)}
        _, state = noise_fn(_paired(grads), state)
        assert state._step_counter == 1
        _, state = noise_fn(_paired(grads), state)
        assert state._step_counter == 2

    def test_deterministic_with_same_key(self, grad_template):
        strategy = band_mf_strategy(n_steps=50, bands=5, momentum=0.9)
        second_strategy = band_mf_strategy(n_steps=50, bands=5, momentum=0.99)
        grads = {"w": torch.randn(4, 3), "b": torch.randn(4)}

        noise_fn1, state1 = mf_noise(
            grad_template,
            strategy,
            noise_multiplier=1.0,
            key=key(42),
            second_moment_strategy=second_strategy,
        )
        output1, _ = noise_fn1(_paired(grads), state1)

        noise_fn2, state2 = mf_noise(
            grad_template,
            strategy,
            noise_multiplier=1.0,
            key=key(42),
            second_moment_strategy=second_strategy,
        )
        output2, _ = noise_fn2(_paired(grads), state2)

        torch.testing.assert_close(
            output1.noisy_grads.pytree["w"], output2.noisy_grads.pytree["w"]
        )
        torch.testing.assert_close(
            output1.noisy_squared_grads.pytree["w"],
            output2.noisy_squared_grads.pytree["w"],
        )

    def test_different_keys_give_different_noise(self, grad_template):
        strategy = band_mf_strategy(n_steps=50, bands=5, momentum=0.9)
        second_strategy = band_mf_strategy(n_steps=50, bands=5, momentum=0.99)
        grads = {"w": torch.randn(4, 3), "b": torch.randn(4)}

        noise_fn1, state1 = mf_noise(
            grad_template,
            strategy,
            noise_multiplier=1.0,
            key=key(42),
            second_moment_strategy=second_strategy,
        )
        output1, _ = noise_fn1(_paired(grads), state1)

        noise_fn2, state2 = mf_noise(
            grad_template,
            strategy,
            noise_multiplier=1.0,
            key=key(99),
            second_moment_strategy=second_strategy,
        )
        output2, _ = noise_fn2(_paired(grads), state2)

        assert not torch.allclose(
            output1.noisy_grads.pytree["w"], output2.noisy_grads.pytree["w"]
        )

    @pytest.mark.parametrize("mechanism", ["band_mf", "blt", "bisr", "bsr", "identity"])
    def test_works_with_supported_mechanisms(self, grad_template, mechanism):
        if mechanism == "band_mf":
            strategy = band_mf_strategy(n_steps=50, bands=5, momentum=0.9)
            second_strategy = band_mf_strategy(n_steps=50, bands=5, momentum=0.99)
        elif mechanism == "blt":
            strategy = blt_strategy(n_steps=50, min_sep=50, momentum=0.9)
            second_strategy = blt_strategy(n_steps=50, min_sep=50, momentum=0.99)
        elif mechanism == "bisr":
            strategy = bisr_strategy(bandwidth=4, n_steps=50, min_sep=50, momentum=0.9)
            second_strategy = bisr_strategy(
                bandwidth=4, n_steps=50, min_sep=50, momentum=0.99
            )
        elif mechanism == "bsr":
            strategy = bsr_strategy(
                bandwidth=4,
                n_steps=50,
                min_sep=50,
                alpha=1.0,
                beta=0.9,
            )
            second_strategy = bsr_strategy(
                bandwidth=4,
                n_steps=50,
                min_sep=50,
                alpha=1.0,
                beta=0.99,
            )
        elif mechanism == "identity":
            strategy = identity_strategy()
            second_strategy = identity_strategy()

        noise_fn, state = mf_noise(
            grad_template,
            strategy,
            noise_multiplier=1.0,
            key=key(42),
            second_moment_strategy=second_strategy,
        )
        grads = {"w": torch.randn(4, 3), "b": torch.randn(4)}
        output, new_state = noise_fn(_paired(grads), state)
        assert output.noisy_grads.pytree["w"].shape == (4, 3)
        assert output.noisy_squared_grads.pytree["w"].shape == (4, 3)
        assert isinstance(new_state, SecondMomentMFNoiseState)

    def test_paired_input_requires_second_moment_strategy(self, grad_template):
        """Single-stream mf_noise rejects paired-stream input."""
        strategy = lambda_cgd_strategy(0.9, n_steps=50, min_sep=50)
        noise_fn, state = mf_noise(
            grad_template,
            strategy,
            noise_multiplier=1.0,
            key=key(42),
        )
        grads = {"w": torch.randn(4, 3), "b": torch.randn(4)}
        with pytest.raises(TypeError, match="second_moment_strategy"):
            noise_fn(_paired(grads), state)

    def test_single_input_rejected_when_second_moment_strategy_supplied(
        self, grad_template
    ):
        """Paired-stream mf_noise rejects single-stream input."""
        strategy = lambda_cgd_strategy(0.9, n_steps=50, min_sep=50)
        second_strategy = lambda_cgd_strategy(0.999, n_steps=50, min_sep=50)
        noise_fn, state = mf_noise(
            grad_template,
            strategy,
            noise_multiplier=1.0,
            key=key(42),
            second_moment_strategy=second_strategy,
        )
        grads = {"w": torch.randn(4, 3), "b": torch.randn(4)}
        with pytest.raises(TypeError, match="SecondMomentClippingOutput"):
            noise_fn(_clipped(grads), state)

    def test_lambda_cgd_accepts_explicit_second_strategy(self, grad_template):
        strategy = lambda_cgd_strategy(0.9, n_steps=50, min_sep=50)
        second_strategy = lambda_cgd_strategy(0.999, n_steps=50, min_sep=50)
        noise_fn, state = mf_noise(
            grad_template,
            strategy,
            noise_multiplier=1.0,
            key=key(42),
            second_moment_strategy=second_strategy,
        )
        grads = {"w": torch.randn(4, 3), "b": torch.randn(4)}
        output, _ = noise_fn(_paired(grads), state)
        assert output.noisy_squared_grads.pytree["w"].shape == (4, 3)

    def test_squared_grads_are_noised_not_raw(self, grad_template):
        strategy = band_mf_strategy(n_steps=50, bands=5, momentum=0.9)
        second_strategy = band_mf_strategy(n_steps=50, bands=5, momentum=0.99)
        noise_fn, state = mf_noise(
            grad_template,
            strategy,
            noise_multiplier=1.0,
            key=key(42),
            second_moment_strategy=second_strategy,
        )
        grads = {"w": torch.ones(4, 3), "b": torch.ones(4)}
        output, _ = noise_fn(_paired(grads), state)
        raw_sq = grads["w"] ** 2
        assert not torch.allclose(
            output.noisy_squared_grads.pytree["w"], raw_sq, atol=1e-6
        )
