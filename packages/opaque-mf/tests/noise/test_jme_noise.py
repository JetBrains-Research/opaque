"""Tests for jme_noise() and JME calibration helpers."""

import math

import pytest
import torch

from opaque.noise.mf import (
    band_mf_strategy,
    bisr_strategy,
    bsr_strategy,
    blt_strategy,
    identity_strategy,
    jme_joint_sensitivity,
    jme_lambda,
    jme_noise,
    jme_second_moment_stddev,
    lambda_cgd_strategy,
    JmeNoiseOutput,
    JmeNoiseState,
)
from opaque.random import key


# ── Calibration helpers ──────────────────────────────────────────────


class TestJmeLambda:
    def test_same_strategy(self):
        # C₁ = C₂ → λ = 1/(c_d·ζ²)
        lam = jme_lambda(1.5, 1.5, 0.1)
        assert lam == pytest.approx(1.0 / (2.0 * 0.01), rel=1e-10)

    def test_different_strategies(self):
        lam = jme_lambda(2.0, 1.0, 0.5)
        expected = 4.0 / (2.0 * 0.25 * 1.0)
        assert lam == pytest.approx(expected, rel=1e-10)

    def test_d1(self):
        c1 = 8.0 / (11.0 + 5.0 * math.sqrt(5.0))
        lam = jme_lambda(1.0, 1.0, 1.0, d=1)
        assert lam == pytest.approx(1.0 / c1, rel=1e-10)

    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            jme_lambda(0.0, 1.0, 1.0)
        with pytest.raises(ValueError):
            jme_lambda(1.0, 0.0, 1.0)
        with pytest.raises(ValueError):
            jme_lambda(1.0, 1.0, 0.0)


class TestJmeJointSensitivity:
    def test_d2_formula(self):
        s = jme_joint_sensitivity(1.5, 0.1)
        expected = 0.1 * 1.5 * math.sqrt(1.5)
        assert s == pytest.approx(expected, rel=1e-10)

    def test_ratio_to_first_only(self):
        s = jme_joint_sensitivity(3.0, 0.5)
        first_only = 0.5 * 3.0
        assert s / first_only == pytest.approx(math.sqrt(1.5), rel=1e-10)

    def test_d1(self):
        c1 = 8.0 / (11.0 + 5.0 * math.sqrt(5.0))
        s = jme_joint_sensitivity(2.0, 0.5, d=1)
        expected = 0.5 * 2.0 * math.sqrt(1.0 + 1.0 / c1)
        assert s == pytest.approx(expected, rel=1e-10)

    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            jme_joint_sensitivity(0.0, 1.0)
        with pytest.raises(ValueError):
            jme_joint_sensitivity(1.0, 0.0)


class TestJmeSecondMomentStddev:
    def test_basic(self):
        assert jme_second_moment_stddev(1.0, 4.0) == pytest.approx(0.5)
        assert jme_second_moment_stddev(1.0, 1.0) == pytest.approx(1.0)

    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            jme_second_moment_stddev(1.0, 0.0)


# ── jme_noise() ─────────────────────────────────────────────────────


class TestJmeNoise:
    @pytest.fixture
    def grad_template(self):
        return {"w": torch.zeros(4, 3), "b": torch.zeros(4)}

    def test_returns_correct_types(self, grad_template):
        strategy = band_mf_strategy(n_steps=50, bands=5, momentum=0.9)
        noise_fn, state = jme_noise(
            grad_template,
            strategy,
            noise_multiplier=1.0,
            key=key(42),
            zeta=0.1,
        )
        assert isinstance(state, JmeNoiseState)

        grads = {"w": torch.randn(4, 3), "b": torch.randn(4)}
        output, new_state = noise_fn(grads, state)
        assert isinstance(output, JmeNoiseOutput)
        assert isinstance(new_state, JmeNoiseState)

    def test_output_shapes_match_input(self, grad_template):
        strategy = band_mf_strategy(n_steps=50, bands=5, momentum=0.9)
        noise_fn, state = jme_noise(
            grad_template,
            strategy,
            noise_multiplier=1.0,
            key=key(42),
            zeta=0.1,
        )
        grads = {"w": torch.randn(4, 3), "b": torch.randn(4)}
        (noisy_g, noisy_sq), state = noise_fn(grads, state)
        assert noisy_g["w"].shape == (4, 3)
        assert noisy_g["b"].shape == (4,)
        assert noisy_sq["w"].shape == (4, 3)
        assert noisy_sq["b"].shape == (4,)

    def test_tuple_unpacking(self, grad_template):
        strategy = band_mf_strategy(n_steps=50, bands=5, momentum=0.9)
        noise_fn, state = jme_noise(
            grad_template,
            strategy,
            noise_multiplier=1.0,
            key=key(42),
            zeta=0.1,
        )
        grads = {"w": torch.randn(4, 3), "b": torch.randn(4)}
        (noisy_g, noisy_sq), new_state = noise_fn(grads, state)
        assert isinstance(noisy_g, dict)
        assert isinstance(noisy_sq, dict)

    def test_step_counter_increments(self, grad_template):
        strategy = band_mf_strategy(n_steps=50, bands=5, momentum=0.9)
        noise_fn, state = jme_noise(
            grad_template,
            strategy,
            noise_multiplier=1.0,
            key=key(42),
            zeta=0.1,
        )
        assert state._step_counter == 0
        grads = {"w": torch.randn(4, 3), "b": torch.randn(4)}
        _, state = noise_fn(grads, state)
        assert state._step_counter == 1
        _, state = noise_fn(grads, state)
        assert state._step_counter == 2

    def test_deterministic_with_same_key(self, grad_template):
        strategy = band_mf_strategy(n_steps=50, bands=5, momentum=0.9)
        grads = {"w": torch.randn(4, 3), "b": torch.randn(4)}

        noise_fn1, state1 = jme_noise(
            grad_template,
            strategy,
            noise_multiplier=1.0,
            key=key(42),
            zeta=0.1,
        )
        (g1, sq1), _ = noise_fn1(grads, state1)

        noise_fn2, state2 = jme_noise(
            grad_template,
            strategy,
            noise_multiplier=1.0,
            key=key(42),
            zeta=0.1,
        )
        (g2, sq2), _ = noise_fn2(grads, state2)

        torch.testing.assert_close(g1["w"], g2["w"])
        torch.testing.assert_close(sq1["w"], sq2["w"])

    def test_different_keys_give_different_noise(self, grad_template):
        strategy = band_mf_strategy(n_steps=50, bands=5, momentum=0.9)
        grads = {"w": torch.randn(4, 3), "b": torch.randn(4)}

        noise_fn1, s1 = jme_noise(
            grad_template,
            strategy,
            noise_multiplier=1.0,
            key=key(42),
            zeta=0.1,
        )
        (g1, _), _ = noise_fn1(grads, s1)

        noise_fn2, s2 = jme_noise(
            grad_template,
            strategy,
            noise_multiplier=1.0,
            key=key(99),
            zeta=0.1,
        )
        (g2, _), _ = noise_fn2(grads, s2)

        assert not torch.allclose(g1["w"], g2["w"])

    @pytest.mark.parametrize("mechanism", ["band_mf", "blt", "bisr", "bsr", "identity"])
    def test_works_with_all_mechanisms(self, grad_template, mechanism):
        if mechanism == "band_mf":
            strategy = band_mf_strategy(n_steps=50, bands=5, momentum=0.9)
        elif mechanism == "blt":
            strategy = blt_strategy(n_steps=50, min_sep=50, momentum=0.9)
        elif mechanism == "bisr":
            strategy = bisr_strategy(bandwidth=4, n_steps=50, min_sep=50)
        elif mechanism == "bsr":
            strategy = bsr_strategy(
                bandwidth=4,
                n_steps=50,
                min_sep=50,
                alpha=1.0,
                beta=0.9,
            )
        elif mechanism == "identity":
            strategy = identity_strategy()

        noise_fn, state = jme_noise(
            grad_template,
            strategy,
            noise_multiplier=1.0,
            key=key(42),
            zeta=0.1,
        )
        grads = {"w": torch.randn(4, 3), "b": torch.randn(4)}
        (noisy_g, noisy_sq), new_state = noise_fn(grads, state)
        assert noisy_g["w"].shape == (4, 3)
        assert noisy_sq["w"].shape == (4, 3)

    def test_band_mf_derives_second_stream_with_same_lr_schedule(self, grad_template):
        lr = torch.linspace(0.001, 0.01, 50, dtype=torch.float64)
        strategy = band_mf_strategy(n_steps=50, bands=5, momentum=0.9, lr_schedule=lr)
        noise_fn, state = jme_noise(
            grad_template,
            strategy,
            noise_multiplier=1.0,
            key=key(42),
            zeta=0.1,
            beta2=0.99,
        )
        grads = {"w": torch.randn(4, 3), "b": torch.randn(4)}
        (noisy_g, noisy_sq), _ = noise_fn(grads, state)
        assert noisy_g["w"].shape == (4, 3)
        assert noisy_sq["w"].shape == (4, 3)

    def test_lambda_cgd_requires_explicit_second_strategy(self, grad_template):
        strategy = lambda_cgd_strategy(0.9, n_steps=50, min_sep=50)
        with pytest.raises(ValueError, match="LambdaCgdStrategy"):
            jme_noise(
                grad_template,
                strategy,
                noise_multiplier=1.0,
                key=key(42),
                zeta=0.1,
            )

    def test_squared_grads_are_noised_not_raw(self, grad_template):
        """The noisy squared grads should differ from raw g²."""
        strategy = band_mf_strategy(n_steps=50, bands=5, momentum=0.9)
        noise_fn, state = jme_noise(
            grad_template,
            strategy,
            noise_multiplier=1.0,
            key=key(42),
            zeta=0.1,
        )
        grads = {"w": torch.ones(4, 3), "b": torch.ones(4)}
        (_, noisy_sq), _ = noise_fn(grads, state)
        raw_sq = grads["w"] ** 2
        assert not torch.allclose(noisy_sq["w"], raw_sq, atol=1e-6)
