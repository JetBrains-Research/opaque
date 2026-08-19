"""Provider-free calibration tests for private second-moment MF noise."""

import math

import pytest

from opaque.api.engine.noise_allocation import paired_noise_stddevs


class TestSecondMomentCalibration:
    def test_paired_stddevs_with_strategy_norms(self):
        zeta, c1, c2 = 0.2, 2.0, 1.5
        noise_multiplier = 3.0
        delta1 = zeta * c1
        delta2 = zeta**2 * c2
        first, second = paired_noise_stddevs(
            noise_multiplier, first=delta1, second=delta2
        )
        total = delta1 + delta2
        assert first == pytest.approx(noise_multiplier * math.sqrt(delta1 * total))
        assert second == pytest.approx(noise_multiplier * math.sqrt(delta2 * total))

    def test_mahalanobis_equality(self):
        delta1 = 0.5 * 2.0
        delta2 = 0.5**2 * 1.0
        first, second = paired_noise_stddevs(1.0, first=delta1, second=delta2)
        mahalanobis = (delta1 / first) ** 2 + (delta2 / second) ** 2
        assert mahalanobis == pytest.approx(1.0, rel=1e-12)

    def test_squared_max_norm_couples_both_streams(self):
        a_first, a_second = paired_noise_stddevs(1.0, first=0.1, second=0.01)
        b_first, b_second = paired_noise_stddevs(1.0, first=0.1, second=0.04)
        assert b_first > a_first
        assert b_second > a_second

    def test_rejects_invalid(self):
        with pytest.raises(ValueError, match="noise_multiplier must be non-negative"):
            paired_noise_stddevs(-1.0, first=0.1, second=0.01)
        with pytest.raises(ValueError, match="first must be non-negative"):
            paired_noise_stddevs(1.0, first=-0.1, second=0.01)
        with pytest.raises(ValueError, match="second must be non-negative"):
            paired_noise_stddevs(1.0, first=0.1, second=-0.01)
