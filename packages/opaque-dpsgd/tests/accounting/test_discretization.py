"""Tests for opaque.accounting.discretization — module-level PLD config management."""

import pytest

import opaque.accounting as acc
import opaque.dpsgd.accounting as dpsgd_acc
from opaque.api.accounting.core.discretization import (
    DiscretizationConfig,
    get_discretization,
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

    def test_default_returns_library_default(self):
        """Before set_discretization(), returns library default (1e-4)."""
        from opaque.accounting import discretization

        discretization._default_config = None
        cfg = get_discretization()
        assert cfg is not None
        assert cfg.discretization == pytest.approx(1e-4)  # Library default
        assert cfg.log_x_mass_truncation_bound == pytest.approx(
            -50.0
        )  # Library default
        assert cfg.max_grid_size == 10_000_000

    def test_set_and_get(self):
        set_discretization(discretization=1e-3)
        cfg = get_discretization()
        assert cfg is not None
        assert cfg.discretization == pytest.approx(1e-3)

    def test_set_all_params(self):
        set_discretization(
            discretization=1e-3,
            log_x_mass_truncation_bound=-40.0,
            max_grid_size=500_000,
            tail_mass_truncation=1e-12,
            num_mc_samples=10_000,
            seed=7,
        )
        cfg = get_discretization()
        assert cfg.discretization == pytest.approx(1e-3)
        assert cfg.log_x_mass_truncation_bound == pytest.approx(-40.0)
        assert cfg.max_grid_size == 500_000
        assert cfg.tail_mass_truncation == pytest.approx(1e-12)
        assert cfg.num_mc_samples == 10_000
        assert cfg.seed == 7

    def test_overwrite(self):
        set_discretization(discretization=1e-3)
        set_discretization(discretization=1e-5)
        cfg = get_discretization()
        assert cfg.discretization == pytest.approx(1e-5)


class TestDiscretizationAffectsResults:
    """Changing discretization affects computed privacy metrics."""

    def test_coarser_grid_changes_epsilon(self):
        """Coarser discretization should produce a different (less precise) epsilon."""
        proc = dpsgd_acc.gaussian(0.8)
        eps_fine = proc.epsilon_at(1e-5, discretization=1e-4)
        eps_coarse = proc.epsilon_at(1e-5, discretization=2e-1)
        # Both valid, but coarser grid = different result
        assert eps_fine != pytest.approx(eps_coarse, rel=1e-3)


class TestQueryTimeOverrides:
    """Test query-time parameter overrides via get_discretization()."""

    def test_discretization_override(self):
        """Query-time discretization override takes precedence."""

        # Set global default
        set_discretization(discretization=1e-3)

        # Query-time override should take precedence
        cfg = get_discretization(discretization=1e-5)
        assert cfg.discretization == pytest.approx(1e-5)

        # Global default unchanged
        global_cfg = get_discretization()
        assert global_cfg.discretization == pytest.approx(1e-3)

    def test_log_x_mass_truncation_bound_override(self):
        """Query-time log_x_mass_truncation_bound override works."""
        # Set global default
        set_discretization(log_x_mass_truncation_bound=-50.0)

        # Query-time override
        cfg = get_discretization(log_x_mass_truncation_bound=-30.0)
        assert cfg.log_x_mass_truncation_bound == pytest.approx(-30.0)

        # Global default unchanged
        global_cfg = get_discretization()
        assert global_cfg.log_x_mass_truncation_bound == pytest.approx(-50.0)

    def test_max_grid_size_override(self):
        """Query-time max_grid_size override works."""
        set_discretization(max_grid_size=10_000_000)

        cfg = get_discretization(max_grid_size=5_000_000)
        assert cfg.max_grid_size == 5_000_000

        global_cfg = get_discretization()
        assert global_cfg.max_grid_size == 10_000_000

    def test_multiple_overrides(self):
        """Multiple query-time overrides work together."""
        set_discretization(
            discretization=1e-3,
            log_x_mass_truncation_bound=-50.0,
            max_grid_size=10_000_000,
        )

        cfg = get_discretization(
            discretization=1e-5,
            log_x_mass_truncation_bound=-30.0,
        )

        # Overridden values
        assert cfg.discretization == pytest.approx(1e-5)
        assert cfg.log_x_mass_truncation_bound == pytest.approx(-30.0)

        # Non-overridden values from global default
        assert cfg.max_grid_size == 10_000_000

    def test_override_on_library_default(self):
        """Query-time override works even without set_discretization."""
        from opaque.accounting import discretization

        discretization._default_config = None  # Clear global default

        cfg = get_discretization(
            discretization=1e-5,
            log_x_mass_truncation_bound=-30.0,
        )

        assert cfg.discretization == pytest.approx(1e-5)
        assert cfg.log_x_mass_truncation_bound == pytest.approx(-30.0)
        # Library defaults for rest
        assert cfg.max_grid_size == 10_000_000


class TestDiscretizationBehavior:
    """Native configuration conversion and exact-grid behavior."""

    def test_config_converts_native_controls(self):
        config = DiscretizationConfig(
            discretization=1e-3,
            log_x_mass_truncation_bound=-40.0,
            max_grid_size=500_000,
            tail_mass_truncation=1e-12,
            num_mc_samples=10_000,
            seed=7,
        )

        native = config.to_native()
        assert native.discretization == pytest.approx(1e-3)
        assert native.log_mass_truncation_bound == pytest.approx(-40.0)
        assert native.max_grid_size == 500_000
        assert native.tail_mass_truncation == pytest.approx(1e-12)
        assert native.num_mc_samples == 10_000
        assert native.seed == 7
        assert config == DiscretizationConfig(
            discretization=1e-3,
            log_x_mass_truncation_bound=-40.0,
            max_grid_size=500_000,
            tail_mass_truncation=1e-12,
            num_mc_samples=10_000,
            seed=7,
        )
        assert hash(config) == hash(config)

    def test_exact_atoms_and_composition_round_up_to_the_grid(self):
        process = acc.eps_delta(0.11, 0.0)
        pld = process.pld(discretization=0.1)

        assert pld.delta_at(0.11) > 0.0
        assert pld.delta_at(0.2) == pytest.approx(0.0)
        composed = pld.compose(pld)
        assert composed.delta_at(0.22) > 0.0
        assert composed.delta_at(0.4) == pytest.approx(0.0)
