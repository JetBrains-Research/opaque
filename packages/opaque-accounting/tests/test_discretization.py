"""Tests for DiscretizationConfig."""

import pytest

import opaque_accounting as acc
from opaque_accounting import DiscretizationConfig


class TestDiscretizationConfig:
    """Test DiscretizationConfig construction and defaults."""

    def test_default_values(self):
        cfg = DiscretizationConfig()
        assert cfg.discretization == pytest.approx(1e-4)
        assert cfg.log_x_mass_truncation_bound == pytest.approx(-50.0)
        assert cfg.pessimistic_estimate is True
        assert cfg.max_grid_size == 10_000_000

    def test_custom_values(self):
        cfg = DiscretizationConfig(
            discretization=1e-3,
            log_x_mass_truncation_bound=-40.0,
            pessimistic_estimate=False,
            max_grid_size=500_000,
        )
        assert cfg.discretization == pytest.approx(1e-3)
        assert cfg.log_x_mass_truncation_bound == pytest.approx(-40.0)
        assert cfg.pessimistic_estimate is False
        assert cfg.max_grid_size == 500_000

    def test_frozen(self):
        cfg = DiscretizationConfig()
        with pytest.raises(AttributeError):
            cfg.discretization = 1e-3


class TestDiscretizationAffectsResults:
    """Changing discretization affects computed privacy metrics."""

    def test_coarser_grid_changes_epsilon(self):
        """Coarser discretization should produce a different (less precise) epsilon."""
        proc = acc.gaussian(0.8)
        eps_fine = proc.pmf(DiscretizationConfig(discretization=1e-4)).epsilon_at(1e-5)
        eps_coarse = proc.pmf(DiscretizationConfig(discretization=2e-1)).epsilon_at(1e-5)
        # Both valid, but coarser grid = different result
        assert eps_fine != pytest.approx(eps_coarse, rel=1e-3)
