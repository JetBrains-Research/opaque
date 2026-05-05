"""Tests for DP-lambda-CGD noise generation via PRNG replay."""

import pytest
import torch

import opaque.accounting as acc
from opaque.types import clipped

from opaque.types import NoisedPytree

from opaque.dpftrl.noise.lambda_cgd import LambdaCgdStrategy, lambda_cgd_strategy
from opaque.dpftrl.noise import mf_noise
from opaque.random import key


def _make_noise(template, n_steps=100, lambda_=0.9, normalized=True, seed=42):
    """Helper: create lambda-CGD noise via the strategy + mf_noise API.

    Uses ``noise_multiplier=1.0`` so realized stddev equals each call's
    ``ClippedPytree.max_norm``; tests pass ``max_norm=1.0`` to recover the
    historical ``stddev=1.0`` semantics.
    """
    strategy = lambda_cgd_strategy(
        lambda_,
        n_steps=n_steps,
        min_sep=1,
        max_participations=1,
        normalized=normalized,
    )
    return mf_noise(template, strategy, noise_multiplier=1.0, key=key(seed))


def _call(noise_fn, grad_pytree, state, *, max_norm=1.0):
    """Wrap ``grad_pytree`` as clipped, run noise, return (noisy_pytree, state)."""
    noisy_out, new_state = noise_fn(clipped(grad_pytree, max_norm=max_norm), state)
    assert isinstance(noisy_out, NoisedPytree)
    return noisy_out.pytree, new_state


class TestLambdaCgdNoise:
    def _make_template(self):
        return {"w": torch.zeros(10)}

    def test_basic_noise_generation(self):
        """Noise function returns correctly shaped output."""
        template = self._make_template()
        noise_fn, state = _make_noise(template)
        noised, new_state = _call(noise_fn, {"w": torch.zeros(10)}, state)
        assert noised["w"].shape == (10,)
        assert new_state._step_counter == 1

    def test_deterministic_with_same_key(self):
        """Same key produces identical noise sequences."""
        template = self._make_template()
        results = []
        for _ in range(2):
            noise_fn, state = _make_noise(template)
            noised, state = _call(noise_fn, {"w": torch.zeros(10)}, state)
            noisy2, state = _call(noise_fn, {"w": torch.zeros(10)}, state)
            results.append(torch.cat([noised["w"], noisy2["w"]]))
        torch.testing.assert_close(results[0], results[1])

    def test_different_keys_give_different_noise(self):
        """Different keys produce different sequences."""
        template = self._make_template()
        noise_fn1, state1 = _make_noise(template, seed=1)
        noise_fn2, state2 = _make_noise(template, seed=2)
        noisy1, _ = _call(noise_fn1, {"w": torch.zeros(10)}, state1)
        noisy2, _ = _call(noise_fn2, {"w": torch.zeros(10)}, state2)
        assert not torch.allclose(noisy1["w"], noisy2["w"])

    def test_lambda_zero_is_independent(self):
        """lambda=0 should produce independent noise at each step (DP-SGD)."""
        template = self._make_template()
        noise_fn, state = _make_noise(template, lambda_=0.0)

        noisy0, state = _call(noise_fn, {"w": torch.zeros(10)}, state)
        noisy1, state = _call(noise_fn, {"w": torch.zeros(10)}, state)

        assert noisy0["w"].std().item() > 0.1
        assert noisy1["w"].std().item() > 0.1

    def test_first_step_same_as_lambda_zero(self):
        """First step is z_0 regardless of lambda (no previous noise) -- unnormalized."""
        template = self._make_template()

        noise_fn_corr, state_corr = _make_noise(template, lambda_=0.9, normalized=False)
        noise_fn_ind, state_ind = _make_noise(template, lambda_=0.0, normalized=False)

        noisy_corr, _ = _call(noise_fn_corr, {"w": torch.zeros(10)}, state_corr)
        noisy_ind, _ = _call(noise_fn_ind, {"w": torch.zeros(10)}, state_ind)

        torch.testing.assert_close(noisy_corr["w"], noisy_ind["w"])

    def test_correlation_changes_second_step(self):
        """Second step with lambda>0 should differ from lambda=0."""
        template = self._make_template()

        noise_fn_corr, state_corr = _make_noise(template, lambda_=0.9)
        noise_fn_ind, state_ind = _make_noise(template, lambda_=0.0)

        _, state_corr = _call(noise_fn_corr, {"w": torch.zeros(10)}, state_corr)
        _, state_ind = _call(noise_fn_ind, {"w": torch.zeros(10)}, state_ind)

        noisy_corr, _ = _call(noise_fn_corr, {"w": torch.zeros(10)}, state_corr)
        noisy_ind, _ = _call(noise_fn_ind, {"w": torch.zeros(10)}, state_ind)

        assert not torch.allclose(noisy_corr["w"], noisy_ind["w"])

    def test_multi_param(self):
        """Works with multiple parameter tensors."""
        template = {"w1": torch.zeros(5), "w2": torch.zeros(3, 4)}
        noise_fn, state = _make_noise(template)
        noised, new_state = _call(
            noise_fn,
            {"w1": torch.zeros(5), "w2": torch.zeros(3, 4)},
            state,
        )
        assert noised["w1"].shape == (5,)
        assert noised["w2"].shape == (3, 4)

    def test_noise_adds_to_grads(self):
        """Noise is added to the gradient, not overwriting it."""
        template = self._make_template()
        noise_fn, state = _make_noise(template)
        grad = {"w": torch.ones(10) * 5.0}
        noised, _ = _call(noise_fn, grad, state)
        noise_fn2, state2 = _make_noise(template)
        noise_only, _ = _call(noise_fn2, {"w": torch.zeros(10)}, state2)
        torch.testing.assert_close(noised["w"] - 5.0, noise_only["w"])

    def test_step_counter_increments(self):
        """Step counter increments with each call."""
        template = self._make_template()
        noise_fn, state = _make_noise(template)
        assert state._step_counter == 0
        _, state = _call(noise_fn, {"w": torch.zeros(10)}, state)
        assert state._step_counter == 1
        _, state = _call(noise_fn, {"w": torch.zeros(10)}, state)
        assert state._step_counter == 2

    def test_rejects_invalid_lambda(self):
        with pytest.raises(ValueError):
            lambda_cgd_strategy(-0.1, n_steps=100, min_sep=1)
        with pytest.raises(ValueError):
            lambda_cgd_strategy(1.0, n_steps=100, min_sep=1)

    def test_prng_replay_correctness(self):
        """Verify the PRNG replay: step 1's z_prev should equal step 0's z_current."""
        template = {"w": torch.zeros(20)}

        # Run with lambda=0, normalized=False to get individual z_0 and z_1
        noise_fn_ind, state_ind = _make_noise(
            template, n_steps=100, lambda_=0.0, normalized=False
        )
        z0, state_ind = _call(noise_fn_ind, {"w": torch.zeros(20)}, state_ind)
        z1, _ = _call(noise_fn_ind, {"w": torch.zeros(20)}, state_ind)

        # Run with lambda>0, normalized=False: step 1 should be z_1 - lambda*z_0
        noise_fn_corr, state_corr = _make_noise(
            template, n_steps=100, lambda_=0.5, normalized=False
        )
        _, state_corr = _call(noise_fn_corr, {"w": torch.zeros(20)}, state_corr)
        step1_corr, _ = _call(noise_fn_corr, {"w": torch.zeros(20)}, state_corr)

        expected = z1["w"] - 0.5 * z0["w"]
        torch.testing.assert_close(step1_corr["w"], expected, atol=1e-6, rtol=1e-6)


class TestLambdaCgdStrategy:
    def test_returns_correct_type(self):
        s = lambda_cgd_strategy(0.9, n_steps=100, min_sep=25, max_participations=4)
        assert isinstance(s, LambdaCgdStrategy)

    def test_sensitivity_positive(self):
        s = lambda_cgd_strategy(0.9, n_steps=100, min_sep=25, max_participations=4)
        assert s.sensitivity > 0

    def test_gram_matrix_present(self):
        s = lambda_cgd_strategy(0.9, n_steps=100, min_sep=25, max_participations=4)
        assert s.gram_matrix is not None
        assert len(s.gram_matrix) == 25 * 25

    def test_normalized_single_participation_sensitivity_one(self):
        """Normalized + single participation -> sensitivity = 1.0."""
        s = lambda_cgd_strategy(0.9, n_steps=100, min_sep=1, max_participations=1)
        assert s.sensitivity == pytest.approx(1.0, abs=1e-6)

    def test_momentum_not_accepted(self):
        """lambda_cgd_strategy does not accept momentum (use bisr_strategy for that)."""
        with pytest.raises(TypeError):
            lambda_cgd_strategy(
                0.5, n_steps=100, min_sep=25, max_participations=4, momentum=0.95
            )

    def test_unnormalized(self):
        s = lambda_cgd_strategy(
            0.9, n_steps=100, min_sep=25, max_participations=4, normalized=False
        )
        assert s.sensitivity > 0

    def test_internal_fields(self):
        s = lambda_cgd_strategy(0.9, n_steps=100, min_sep=25, max_participations=4)
        assert s._lambda == pytest.approx(0.9)
        assert s._n_steps == 100
        assert s._normalized is True


class TestLambdaCgdPld:
    delta = 1e-5

    def test_lambda_cgd_pld(self):
        s = lambda_cgd_strategy(0.9, n_steps=100, min_sep=25, max_participations=4)
        eps = acc.lambda_cgd(1.0, sensitivity=s.sensitivity).epsilon_at(self.delta)
        assert eps > 0

    def test_lambda_cgd_bnb(self):
        s = lambda_cgd_strategy(0.9, n_steps=100, min_sep=25, max_participations=4)
        eps = acc.balls_in_bins(
            acc.lambda_cgd(1.0, sensitivity=s.sensitivity, gram_matrix=s.gram_matrix),
            num_bins=25,
            num_epochs=4,
        ).epsilon_at(self.delta)
        assert eps > 0
