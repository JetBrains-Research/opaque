"""Tests for private second-moment MF noise."""

import math

import pytest
import torch

from opaque.dpsgd.noise import paired_noise_stddevs
from opaque.types import clipped
from opaque.types import (
    NoisedPytree,
    SecondMomentClippingOutput,
    SecondMomentNoiseOutput,
)
from opaque.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    bsr_strategy,
    blt_strategy,
    identity_strategy,
    lambda_cgd_strategy,
    mf_noise,
)
from opaque.dpftrl.noise.types import SecondMomentMFNoiseState
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
    grads_clipped = clipped(grads, max_norm=_SENSITIVITY)
    sq_pytree = {k: v * v for k, v in grads.items()}
    sq_clipped = clipped(sq_pytree, max_norm=_SENSITIVITY * _SENSITIVITY)
    return SecondMomentClippingOutput(grads=grads_clipped, squared_grads=sq_clipped)


def _clipped(grads):
    """Wrap raw grad pytree as ClippedPytree at the test's standard max_norm."""
    return clipped(grads, max_norm=_SENSITIVITY)


class TestSecondMomentCalibration:
    """``mf_noise`` consumes :func:`paired_noise_stddevs` for σ allocation.

    The strategy norms enter as multipliers on the per-record bounds:
    ``Δ¹ = ζ · ‖C₁‖``, ``Δ² = ζ² · ‖C₂‖``.  These tests pin the closed
    form on representative inputs.
    """

    def test_paired_stddevs_with_strategy_norms(self):
        # Δ¹ = ζ · c1, Δ² = ζ² · c2.
        zeta, c1, c2 = 0.2, 2.0, 1.5
        nm = 3.0
        delta1 = zeta * c1
        delta2 = (zeta**2) * c2
        s_first, s_second = paired_noise_stddevs(nm, first=delta1, second=delta2)
        s_total = delta1 + delta2
        assert s_first == pytest.approx(nm * math.sqrt(delta1 * s_total))
        assert s_second == pytest.approx(nm * math.sqrt(delta2 * s_total))

    def test_mahalanobis_equality(self):
        zeta, c1, c2, nm = 0.5, 2.0, 1.0, 1.0
        delta1 = zeta * c1
        delta2 = (zeta**2) * c2
        s_first, s_second = paired_noise_stddevs(nm, first=delta1, second=delta2)
        mahal = (delta1 / s_first) ** 2 + (delta2 / s_second) ** 2
        assert mahal == pytest.approx(1.0 / nm**2, rel=1e-12)

    def test_squared_max_norm_couples_both_streams(self):
        """Increasing ``squared_max_norm`` shifts both σ's via S = Δ¹+Δ²."""
        a_first, a_second = paired_noise_stddevs(1.0, first=0.1, second=0.01)
        b_first, b_second = paired_noise_stddevs(1.0, first=0.1, second=0.04)
        assert b_first > a_first
        assert b_second > a_second

    def test_rejects_invalid(self):
        with pytest.raises(ValueError):
            paired_noise_stddevs(-1.0, first=0.1, second=0.01)
        with pytest.raises(ValueError):
            paired_noise_stddevs(1.0, first=-0.1, second=0.01)
        with pytest.raises(ValueError):
            paired_noise_stddevs(1.0, first=0.1, second=-0.01)


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
