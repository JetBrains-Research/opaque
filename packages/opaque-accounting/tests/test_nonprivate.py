"""Tests for NonPrivate mechanism — infinite privacy loss, composition annihilator."""

import math

import pytest

import opaque_accounting as acc
from opaque_accounting.base import DpProcess
from opaque_accounting.mechanisms import Identity, NonPrivate
from opaque_accounting.mechanisms.gaussian import Gaussian

# ── NonPrivate dataclass tests ──────────────────────────────────────


class TestNonPrivateDataclass:
    """NonPrivate frozen dataclass."""

    def test_is_dp_process(self):
        assert isinstance(NonPrivate(), DpProcess)

    def test_is_dataclass(self):
        import dataclasses

        assert dataclasses.is_dataclass(NonPrivate())

    def test_equality(self):
        assert NonPrivate() == NonPrivate()
        assert NonPrivate() != Identity()


class TestNonPrivateConstructor:
    """acc.nonprivate() returns NonPrivate."""

    def test_returns_nonprivate(self):
        n = acc.nonprivate()
        assert isinstance(n, NonPrivate)

    def test_gaussian_zero_returns_gaussian(self):
        """gaussian(0) should return Gaussian with non-private PLD."""
        g = acc.gaussian(0)
        assert isinstance(g, Gaussian)
        assert g.noise_multiplier == 0
        assert g.epsilon_at(1e-5) == math.inf

    def test_truncated_gaussian_zero_returns_truncated_gaussian(self):
        """truncated_gaussian(0) should return TruncatedGaussian with non-private PLD."""
        from opaque_accounting.mechanisms.truncated_gaussian import TruncatedGaussian

        tg = acc.truncated_gaussian(0)
        assert isinstance(tg, TruncatedGaussian)
        assert tg.noise_multiplier == 0
        assert tg.epsilon_at(1e-5) == math.inf

    def test_poisson_gaussian_zero(self):
        """poisson(gaussian(0)) should produce non-private PLD."""
        step = acc.poisson(acc.gaussian(0), sample_rate=0.01)
        assert step.epsilon_at(1e-5) == math.inf

    def test_truncated_poisson_gaussian_zero(self):
        """truncated_poisson(gaussian(0)) should produce non-private PLD."""
        step = acc.truncated_poisson(
            acc.gaussian(0),
            sample_rate=0.01,
            batch_size_cap=128,
            dataset_size=10_000,
        )
        assert step.epsilon_at(1e-5) == math.inf

    def test_adaclip_gaussian_zero(self):
        """adaclip(gaussian(0)) should produce non-private PLD."""
        step = acc.adaclip(acc.gaussian(0), expected_batch_size=100)
        assert step.epsilon_at(1e-5) == math.inf
        assert step.effective_noise_multiplier == 0.0


# ── PLD metrics ─────────────────────────────────────────────────────


class TestNonPrivateMetrics:
    """NonPrivate gives ε=∞, δ=1, advantage=1."""

    def test_epsilon_is_inf(self):
        n = acc.nonprivate()
        for delta in [1e-10, 1e-5, 0.1, 0.5]:
            eps = n.epsilon_at(delta)
            assert eps == math.inf, f"delta={delta}, got eps={eps}"

    def test_delta_is_one(self):
        n = acc.nonprivate()
        for epsilon in [0.0, 0.1, 1.0, 10.0, 100.0]:
            d = n.delta_at(epsilon)
            assert d == pytest.approx(1.0), f"eps={epsilon}, got delta={d}"

    def test_advantage_is_one(self):
        adv = acc.nonprivate().advantage()
        assert adv == pytest.approx(1.0)

    def test_beta_at_zero_alpha(self):
        beta = acc.nonprivate().beta_at(0.0)
        assert beta == pytest.approx(0.0, abs=1e-9)

    def test_risk_at_half(self):
        risk = acc.nonprivate().risk_at(0.5)
        assert risk == pytest.approx(0.0, abs=1e-6)


# ── Combinator support ──────────────────────────────────────────────


class TestNonPrivatePoisson:
    """NonPrivate threads through Poisson."""

    def test_poisson_accepts_nonprivate(self):
        step = acc.poisson(acc.nonprivate(), sample_rate=0.01)
        eps = step.epsilon_at(1e-5)
        assert eps == math.inf

    def test_poisson_delta_is_one(self):
        step = acc.poisson(acc.nonprivate(), sample_rate=0.01)
        d = step.delta_at(1.0)
        assert d == pytest.approx(1.0)


class TestNonPrivateTruncatedPoisson:
    """NonPrivate threads through TruncatedPoisson."""

    def test_truncated_poisson_accepts_nonprivate(self):
        step = acc.truncated_poisson(
            acc.nonprivate(), sample_rate=0.01, batch_size_cap=128, dataset_size=10_000
        )
        eps = step.epsilon_at(1e-5)
        assert eps == math.inf

    def test_truncated_poisson_adaclip_nonprivate(self):
        """Full chain: truncated_poisson(adaclip(nonprivate()))."""
        step = acc.truncated_poisson(
            acc.adaclip(acc.nonprivate(), expected_batch_size=100),
            sample_rate=0.01,
            batch_size_cap=128,
            dataset_size=10_000,
        )
        eps = step.epsilon_at(1e-5)
        assert eps == math.inf


class TestNonPrivateParallelPoisson:
    """NonPrivate threads through ParallelPoisson."""

    def test_parallel_poisson_accepts_nonprivate(self):
        step = acc.parallel_poisson(acc.nonprivate(), sample_rate=0.01, num_workers=4)
        eps = step.epsilon_at(1e-5)
        assert eps == math.inf

    def test_parallel_poisson_adaclip_nonprivate(self):
        """Full chain: parallel_poisson(adaclip(nonprivate()))."""
        step = acc.parallel_poisson(
            acc.adaclip(acc.nonprivate(), expected_batch_size=100),
            sample_rate=0.01,
            num_workers=4,
        )
        eps = step.epsilon_at(1e-5)
        assert eps == math.inf


class TestNonPrivateAdaClip:
    """NonPrivate threads through AdaClip."""

    def test_adaclip_accepts_nonprivate(self):
        step = acc.adaclip(acc.nonprivate(), expected_batch_size=100)
        eps = step.epsilon_at(1e-5)
        assert eps == math.inf

    def test_poisson_adaclip_nonprivate(self):
        """Full chain: poisson(adaclip(nonprivate()))."""
        step = acc.poisson(
            acc.adaclip(acc.nonprivate(), expected_batch_size=100),
            sample_rate=0.01,
        )
        eps = step.epsilon_at(1e-5)
        assert eps == math.inf

    def test_effective_noise_multiplier_returns_zero(self):
        """AdaClip(NonPrivate()).effective_noise_multiplier should return 0.0."""
        ac = acc.adaclip(acc.nonprivate(), expected_batch_size=100)
        assert ac.effective_noise_multiplier == 0.0


# ── Composition ─────────────────────────────────────────────────────


class TestNonPrivateComposition:
    """NonPrivate composes correctly (infinity_mass propagation)."""

    def test_repeat(self):
        training = acc.nonprivate() * 1000
        eps = training.epsilon_at(1e-5)
        assert eps == math.inf

    def test_compose_with_gaussian(self):
        composed = acc.gaussian(1.1) | acc.nonprivate()
        eps = composed.epsilon_at(1e-5)
        assert eps == math.inf

    def test_compose_nonprivate_first(self):
        composed = acc.nonprivate() | acc.gaussian(1.1)
        eps = composed.epsilon_at(1e-5)
        assert eps == math.inf

    def test_compose_with_poisson_step(self):
        """Realistic: private steps composed with a nonprivate step."""
        private_step = acc.poisson(acc.gaussian(1.1), sample_rate=0.01)
        training = (private_step * 100) | acc.nonprivate()
        eps = training.epsilon_at(1e-5)
        assert eps == math.inf

    def test_accountant_accumulates_nonprivate(self):
        """Accountant |= nonprivate() works without guards."""
        accounting = acc.Accountant()
        step = acc.poisson(acc.nonprivate(), sample_rate=0.01)
        for _ in range(10):
            accounting |= step
        eps = accounting.epsilon_at(1e-5)
        assert eps == math.inf

    def test_cached_nonprivate(self):
        """acc.cached() works with nonprivate accounting."""
        accounting = acc.Accountant()
        accounting |= acc.nonprivate()
        accounting = acc.cached(accounting)
        eps = accounting.epsilon_at(1e-5)
        assert eps == math.inf


# ── Serialization ───────────────────────────────────────────────────


class TestNonPrivateSerialization:
    """NonPrivate round-trips through state_dict."""

    def test_state_dict(self):
        n = acc.nonprivate()
        d = n.state_dict()
        assert d["type"] == "NonPrivate"

    def test_round_trip(self):
        n = acc.nonprivate()
        d = n.state_dict()
        restored = DpProcess.from_state_dict(d)
        assert isinstance(restored, NonPrivate)
        assert restored.epsilon_at(1e-5) == math.inf
