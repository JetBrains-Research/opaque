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

    def test_blt_standalone(self):
        import math

        import opaque.dpftrl.accounting as ftrl_acc
        from opaque.dpftrl.noise import blt_strategy

        s = blt_strategy(momentum=1.0)
        eps = ftrl_acc.mf_gaussian(
            1.0, s, n_steps=10, min_sep=10, max_participations=1
        ).epsilon_at(1e-5)
        assert math.isfinite(eps)
        assert eps > 0
