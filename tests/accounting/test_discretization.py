"""Tests for opaque.accounting.discretization — module-level PLD config management."""

import pytest

import opaque.accounting as acc
from opaque.accounting.discretization import (
    PldConfig,
    get_discretization,
    resolve_pld_config,
    set_discretization,
)


@pytest.fixture(autouse=True)
def _reset_discretization():
    """Reset module-level default after each test."""
    from opaque.accounting import discretization

    original = discretization._default_config
    yield
    discretization._default_config = original


class TestSetGetDiscretization:
    """set_discretization / get_discretization roundtrip."""

    def test_default_is_none(self):
        """Before set_discretization(), default is None (use Rust defaults)."""
        from opaque.accounting import discretization

        discretization._default_config = None
        assert get_discretization() is None

    def test_set_and_get(self):
        set_discretization(discretization=1e-3)
        cfg = get_discretization()
        assert cfg is not None
        assert cfg.discretization == pytest.approx(1e-3)

    def test_set_all_params(self):
        set_discretization(
            discretization=1e-3,
            log_mass_truncation_bound=-40.0,
            pessimistic_estimate=False,
            max_grid_size=500_000,
        )
        cfg = get_discretization()
        assert cfg.discretization == pytest.approx(1e-3)
        assert cfg.log_mass_truncation_bound == pytest.approx(-40.0)
        assert cfg.pessimistic_estimate is False
        assert cfg.max_grid_size == 500_000

    def test_overwrite(self):
        set_discretization(discretization=1e-3)
        set_discretization(discretization=1e-5)
        cfg = get_discretization()
        assert cfg.discretization == pytest.approx(1e-5)


class TestResolvePldConfig:
    """resolve_pld_config() dispatches correctly."""

    def test_none_returns_module_default(self):
        """None → module-level default."""
        from opaque.accounting import discretization

        discretization._default_config = None
        assert resolve_pld_config(None) is None

        set_discretization(discretization=1e-3)
        result = resolve_pld_config(None)
        assert result is not None
        assert result.discretization == pytest.approx(1e-3)

    def test_float_creates_config(self):
        """Float value → PldConfig with that discretization."""
        result = resolve_pld_config(1e-3)
        assert isinstance(result, PldConfig)
        assert result.discretization == pytest.approx(1e-3)

    def test_int_creates_config(self):
        """Int value → PldConfig (coerced to float)."""
        result = resolve_pld_config(1)
        assert isinstance(result, PldConfig)
        assert result.discretization == pytest.approx(1.0)

    def test_pldconfig_passthrough(self):
        """PldConfig → returned as-is."""
        cfg = PldConfig(discretization=1e-3)
        result = resolve_pld_config(cfg)
        assert result is cfg


class TestDiscretizationAffectsResults:
    """Changing discretization affects computed privacy metrics."""

    def test_coarser_grid_changes_epsilon(self):
        """Coarser discretization should produce a different (less precise) epsilon."""
        fine = acc.gaussian(0.8, discretization=1e-4)
        coarse = acc.gaussian(0.8, discretization=2e-1)
        eps_fine = fine.epsilon_at(1e-5)
        eps_coarse = coarse.epsilon_at(1e-5)
        # Both valid, but coarser grid = different result
        assert eps_fine != pytest.approx(eps_coarse, rel=1e-3)
