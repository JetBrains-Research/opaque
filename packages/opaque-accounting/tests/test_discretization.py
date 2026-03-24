"""Tests for discretization parameters affecting results."""

import pytest

import opaque_accounting as acc


class TestDiscretizationAffectsResults:
    """Changing discretization affects computed privacy metrics."""

    def test_coarser_grid_changes_epsilon(self):
        """Coarser discretization should produce a different (less precise) epsilon."""
        proc = acc.gaussian(0.8)
        eps_fine = proc.pmf(discretization=1e-4).epsilon_at(1e-5)
        eps_coarse = proc.pmf(discretization=2e-1).epsilon_at(1e-5)
        # Both valid, but coarser grid = different result
        assert eps_fine != pytest.approx(eps_coarse, rel=1e-3)
