"""Torch tests for DP-lambda-CGD noise generation via PRNG replay."""

import torch

from opaque.dpftrl.noise import lambda_cgd_strategy, mf_gaussian_noise
from opaque.random import key
from opaque.types import NoisedPytree, clipped


def _make_noise(template, n_steps=100, lambda_=0.9, normalized=True, seed=42):
    return mf_gaussian_noise(
        template,
        lambda_cgd_strategy(lambda_=lambda_, normalized=normalized),
        n_steps=n_steps,
        min_sep=1,
        max_participations=1,
        noise_multiplier=1.0,
        key=key(seed),
    )


def _call(noise_fn, grad_pytree, state, *, max_norm=1.0):
    noisy_out, new_state = noise_fn(clipped(grad_pytree, max_norm=max_norm), state)
    assert isinstance(noisy_out, NoisedPytree)
    return noisy_out.pytree, new_state


class TestLambdaCgdNoise:
    def _make_template(self):
        return {"w": torch.zeros(10)}

    def test_basic_noise_generation(self):
        noise_fn, state = _make_noise(self._make_template())
        noised, new_state = _call(noise_fn, {"w": torch.zeros(10)}, state)
        assert noised["w"].shape == (10,)
        assert new_state._step_counter == 1

    def test_deterministic_with_same_key(self):
        results = []
        for _ in range(2):
            noise_fn, state = _make_noise(self._make_template())
            noised, state = _call(noise_fn, {"w": torch.zeros(10)}, state)
            noisy2, state = _call(noise_fn, {"w": torch.zeros(10)}, state)
            results.append(torch.cat([noised["w"], noisy2["w"]]))
        torch.testing.assert_close(results[0], results[1])

    def test_keyed_noise_ignores_global_torch_rng_draws(self):
        noise_fn, state = _make_noise(self._make_template())
        expected, _ = _call(noise_fn, {"w": torch.zeros(10)}, state)
        torch.manual_seed(999)
        torch.randn(1000)
        noise_fn, state = _make_noise(self._make_template())
        actual, _ = _call(noise_fn, {"w": torch.zeros(10)}, state)
        torch.testing.assert_close(actual["w"], expected["w"])

    def test_different_keys_give_different_noise(self):
        noise_fn1, state1 = _make_noise(self._make_template(), seed=1)
        noise_fn2, state2 = _make_noise(self._make_template(), seed=2)
        noisy1, _ = _call(noise_fn1, {"w": torch.zeros(10)}, state1)
        noisy2, _ = _call(noise_fn2, {"w": torch.zeros(10)}, state2)
        assert not torch.allclose(noisy1["w"], noisy2["w"])

    def test_lambda_zero_is_independent(self):
        noise_fn, state = _make_noise(self._make_template(), lambda_=0.0)
        noisy0, state = _call(noise_fn, {"w": torch.zeros(10)}, state)
        noisy1, _ = _call(noise_fn, {"w": torch.zeros(10)}, state)
        assert noisy0["w"].std().item() > 0.1
        assert noisy1["w"].std().item() > 0.1

    def test_first_step_same_as_lambda_zero(self):
        corr_fn, corr_state = _make_noise(
            self._make_template(), lambda_=0.9, normalized=False
        )
        iid_fn, iid_state = _make_noise(
            self._make_template(), lambda_=0.0, normalized=False
        )
        correlated, _ = _call(corr_fn, {"w": torch.zeros(10)}, corr_state)
        independent, _ = _call(iid_fn, {"w": torch.zeros(10)}, iid_state)
        torch.testing.assert_close(correlated["w"], independent["w"])

    def test_correlation_changes_second_step(self):
        corr_fn, corr_state = _make_noise(self._make_template(), lambda_=0.9)
        iid_fn, iid_state = _make_noise(self._make_template(), lambda_=0.0)
        _, corr_state = _call(corr_fn, {"w": torch.zeros(10)}, corr_state)
        _, iid_state = _call(iid_fn, {"w": torch.zeros(10)}, iid_state)
        correlated, _ = _call(corr_fn, {"w": torch.zeros(10)}, corr_state)
        independent, _ = _call(iid_fn, {"w": torch.zeros(10)}, iid_state)
        assert not torch.allclose(correlated["w"], independent["w"])

    def test_multi_param(self):
        template = {"w1": torch.zeros(5), "w2": torch.zeros(3, 4)}
        noise_fn, state = _make_noise(template)
        noised, _ = _call(noise_fn, template, state)
        assert noised["w1"].shape == (5,)
        assert noised["w2"].shape == (3, 4)

    def test_noise_adds_to_grads(self):
        template = self._make_template()
        noise_fn, state = _make_noise(template)
        noised, _ = _call(noise_fn, {"w": torch.full((10,), 5.0)}, state)
        noise_fn2, state2 = _make_noise(template)
        noise_only, _ = _call(noise_fn2, {"w": torch.zeros(10)}, state2)
        torch.testing.assert_close(noised["w"] - 5.0, noise_only["w"])

    def test_step_counter_increments(self):
        noise_fn, state = _make_noise(self._make_template())
        assert state._step_counter == 0
        _, state = _call(noise_fn, {"w": torch.zeros(10)}, state)
        assert state._step_counter == 1
        _, state = _call(noise_fn, {"w": torch.zeros(10)}, state)
        assert state._step_counter == 2

    def test_prng_replay_correctness(self):
        template = {"w": torch.zeros(20)}
        iid_fn, iid_state = _make_noise(template, lambda_=0.0, normalized=False)
        z0, iid_state = _call(iid_fn, template, iid_state)
        z1, _ = _call(iid_fn, template, iid_state)
        corr_fn, corr_state = _make_noise(template, lambda_=0.5, normalized=False)
        _, corr_state = _call(corr_fn, template, corr_state)
        correlated, _ = _call(corr_fn, template, corr_state)
        torch.testing.assert_close(
            correlated["w"], z1["w"] - 0.5 * z0["w"], atol=1e-6, rtol=1e-6
        )
