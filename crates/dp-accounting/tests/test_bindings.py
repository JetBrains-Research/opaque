"""PyO3 binding smoke tests for opaque_accounting native module.

Verifies that the Rust→Python bridge works: constructors produce correct
types, PLD methods return finite values, config roundtrips.

Does NOT test arithmetic accuracy — that belongs in tests/accounting/.
"""

import math

import pytest

import opaque_accounting as dp


class TestPldConstructors:
    """PLD constructor functions return Pld objects with finite metrics."""

    def test_gaussian_pld(self):
        pld = dp.gaussian_pld(1.1)
        assert isinstance(pld, dp.Pld)
        eps = pld.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_poisson_gaussian_pld(self):
        pld = dp.poisson_gaussian_pld(0.8, 0.01)
        assert isinstance(pld, dp.Pld)
        eps = pld.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_truncated_poisson_gaussian_pld(self):
        pld = dp.truncated_poisson_gaussian_pld(0.8, 0.01, 128, 10_000)
        assert isinstance(pld, dp.Pld)
        eps = pld.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_accumulated_poisson_gaussian_pld(self):
        pld = dp.accumulated_poisson_gaussian_pld(0.8, 0.01, 4)
        assert isinstance(pld, dp.Pld)
        eps = pld.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_eps_delta_pld(self):
        pld = dp.eps_delta_pld(1.0, 1e-5)
        assert isinstance(pld, dp.Pld)

    def test_identity_pld(self):
        pld = dp.identity_pld()
        eps = pld.epsilon_at(1e-5)
        assert eps == pytest.approx(0.0, abs=1e-10)

    def test_combined_sensitivity(self):
        s = dp.combined_sensitivity(1.1, 50.0)
        assert math.isfinite(s) and s > 0


class TestPldOperators:
    """Composition operators on Pld objects."""

    def test_self_compose(self):
        pld = dp.gaussian_pld(0.8)
        composed = pld.self_compose(10)
        assert isinstance(composed, dp.Pld)
        eps = composed.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_mul_operator(self):
        pld = dp.gaussian_pld(0.8)
        composed = pld * 10
        eps = composed.epsilon_at(1e-5)
        assert math.isfinite(eps)

    def test_rmul_operator(self):
        pld = dp.gaussian_pld(0.8)
        composed = 10 * pld
        eps = composed.epsilon_at(1e-5)
        assert math.isfinite(eps)

    def test_compose(self):
        a = dp.gaussian_pld(0.8)
        b = dp.gaussian_pld(0.5)
        composed = a.compose(b)
        assert isinstance(composed, dp.Pld)
        eps = composed.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_or_operator(self):
        a = dp.gaussian_pld(0.8)
        b = dp.gaussian_pld(0.5)
        composed = a | b
        eps = composed.epsilon_at(1e-5)
        assert math.isfinite(eps)


class TestPldMetrics:
    """All metric methods on Pld return finite values."""

    @pytest.fixture()
    def pld(self):
        base = dp.poisson_gaussian_pld(0.8, 0.01)
        return base.self_compose(100)

    def test_epsilon_at(self, pld):
        eps = pld.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_delta_at(self, pld):
        d = pld.delta_at(1.0)
        assert math.isfinite(d)
        assert 0 <= d <= 1

    def test_advantage(self, pld):
        adv = pld.advantage()
        assert math.isfinite(adv)
        assert 0 <= adv <= 1

    def test_beta_at(self, pld):
        beta = pld.beta_at(0.1)
        assert math.isfinite(beta)
        assert 0 <= beta <= 1

    def test_risk_at(self, pld):
        risk = pld.risk_at(0.5)
        assert math.isfinite(risk)
        assert 0 <= risk <= 0.5


class TestPldConfig:
    """PldConfig (DiscretizationConfig) roundtrips."""

    def test_properties(self):
        cfg = dp.PldConfig(
            discretization=1e-3,
            log_mass_truncation_bound=-50.0,
            pessimistic_estimate=False,
            max_grid_size=1_000_000,
        )
        assert cfg.discretization == 1e-3
        assert cfg.log_mass_truncation_bound == -50.0
        assert cfg.pessimistic_estimate is False
        assert cfg.max_grid_size == 1_000_000

    def test_equality(self):
        a = dp.PldConfig(discretization=1e-3)
        b = dp.PldConfig(discretization=1e-3)
        c = dp.PldConfig(discretization=1e-4)
        assert a == b
        assert a != c

    def test_gaussian_accepts_config(self):
        cfg = dp.PldConfig(discretization=1e-3)
        pld = dp.gaussian_pld(0.8, config=cfg)
        eps = pld.epsilon_at(1e-5)
        assert math.isfinite(eps)


class TestRepr:
    """String representation works."""

    def test_str(self):
        pld = dp.gaussian_pld(0.7)
        s = str(pld)
        assert len(s) > 0

    def test_repr(self):
        pld = dp.gaussian_pld(0.7)
        r = repr(pld)
        assert len(r) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
