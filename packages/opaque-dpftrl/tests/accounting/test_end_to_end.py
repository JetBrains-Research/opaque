"""End-to-end accounting tests for ``opaque.dpftrl.accounting``."""

from __future__ import annotations


class TestEndToEndCalibration:
    """Constructed mechanisms compute valid PLDs through the new namespace."""

    def test_band_mf_poisson(self):
        import math

        import opaque.dpftrl.accounting as ftrl_acc
        from opaque.dpftrl.noise import band_mf_strategy

        strategy = band_mf_strategy(bands=2)
        proc = ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.0, strategy),
            sample_rate=0.01,
            n_steps=20,
        )
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps)
        assert eps > 0

    def test_blt_standalone_uses_full_horizon(self):
        import pytest

        import opaque.dpftrl.accounting as ftrl_acc
        from opaque.api.accounting.core import _native
        from opaque.api.accounting.core.discretization import get_discretization
        from opaque.dpftrl.noise import blt_strategy

        strategy = blt_strategy(max_buffers=3)
        context = {"n_steps": 32, "min_sep": 1, "max_participations": 1}
        coefficients = strategy.coefficients(**context)
        # The first Toeplitz column has the largest norm for one participation.
        sensitivity = coefficients.norm().item()
        reference = _native.gaussian_pld(
            2.0 / sensitivity, get_discretization().to_native()
        ).epsilon_at(1e-5)

        eps = ftrl_acc.mf_gaussian(2.0, strategy, **context).epsilon_at(1e-5)
        one_step = ftrl_acc.mf_gaussian(
            2.0, strategy, n_steps=1, min_sep=1, max_participations=1
        ).epsilon_at(1e-5)
        assert eps == pytest.approx(reference, rel=1e-9)
        assert eps > one_step
