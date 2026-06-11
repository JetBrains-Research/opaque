"""Tests for opaque.accounting.calibration — binary search calibration framework."""

import math

import pytest

import opaque.dpsgd.accounting as dpsgd_acc
from opaque.accounting import calibration as cal
from opaque.api.accounting.core.calibration import (
    AdvantageBudget,
    BetaBudget,
    CalibrateResult,
    DeltaBudget,
    EpsilonBudget,
    RiskBudget,
)

# -- Budget validation -------------------------------------------------------


class TestEpsilonBudget:
    def test_valid(self):
        t = cal.epsilon_budget(3.0, delta=1e-5)
        assert isinstance(t, EpsilonBudget)
        assert t.value == 3.0
        assert "epsilon" in t.name

    def test_decreasing(self):
        """Epsilon decreases with noise → decreasing=True."""
        assert cal.epsilon_budget(3.0, delta=1e-5).decreasing is True

    def test_evaluate(self):
        t = cal.epsilon_budget(3.0, delta=1e-5)
        proc = dpsgd_acc.gaussian(0.8)
        val = t.evaluate(proc)
        assert math.isfinite(val)

    def test_rejects_negative_epsilon(self):
        with pytest.raises(ValueError, match="epsilon"):
            cal.epsilon_budget(-1.0, delta=1e-5)

    def test_rejects_delta_out_of_range(self):
        with pytest.raises(ValueError, match="delta"):
            cal.epsilon_budget(3.0, delta=0.0)
        with pytest.raises(ValueError, match="delta"):
            cal.epsilon_budget(3.0, delta=1.0)


class TestDeltaBudget:
    def test_valid(self):
        t = cal.delta_budget(1e-5, epsilon=3.0)
        assert isinstance(t, DeltaBudget)
        assert t.value == pytest.approx(1e-5)

    def test_decreasing(self):
        assert cal.delta_budget(1e-5, epsilon=3.0).decreasing is True

    def test_rejects_invalid(self):
        with pytest.raises(ValueError, match="delta"):
            cal.delta_budget(0.0, epsilon=3.0)
        with pytest.raises(ValueError, match="epsilon"):
            cal.delta_budget(1e-5, epsilon=-1.0)


class TestAdvantageBudget:
    def test_valid(self):
        t = cal.advantage_budget(0.1)
        assert isinstance(t, AdvantageBudget)
        assert t.value == pytest.approx(0.1)

    def test_decreasing(self):
        assert cal.advantage_budget(0.1).decreasing is True

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError, match="advantage"):
            cal.advantage_budget(0.0)
        with pytest.raises(ValueError, match="advantage"):
            cal.advantage_budget(1.0)


class TestBetaBudget:
    def test_valid(self):
        t = cal.beta_budget(0.05, alpha=0.01)
        assert isinstance(t, BetaBudget)
        assert t.value == pytest.approx(0.05)

    def test_decreasing(self):
        """Beta increases with noise → decreasing=False."""
        assert cal.beta_budget(0.05, alpha=0.01).decreasing is False

    def test_rejects_invalid(self):
        with pytest.raises(ValueError, match="beta"):
            cal.beta_budget(0.0, alpha=0.01)
        with pytest.raises(ValueError, match="alpha"):
            cal.beta_budget(0.5, alpha=0.0)


class TestRiskBudget:
    def test_valid(self):
        t = cal.risk_budget(0.1, prior=0.5)
        assert isinstance(t, RiskBudget)
        assert t.value == pytest.approx(0.1)

    def test_decreasing(self):
        """Risk increases with noise → decreasing=False."""
        assert cal.risk_budget(0.1, prior=0.5).decreasing is False

    def test_rejects_invalid(self):
        with pytest.raises(ValueError, match="risk"):
            cal.risk_budget(0.0, prior=0.5)
        with pytest.raises(ValueError, match="prior"):
            cal.risk_budget(0.1, prior=0.0)


# -- Calibration errors -------------------------------------------------------


class TestCalibrateErrors:
    def _process(self, nm):
        return dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), 0.01) * 1000

    def test_param_min_ge_max(self):
        with pytest.raises(ValueError, match="param_min"):
            cal.calibrate(cal.epsilon_budget(3.0, delta=1e-5), self._process, 0.8, 0.5)

    def test_budget_outside_bracket(self):
        """Target not achievable within bounds → ValueError."""
        with pytest.raises(ValueError):
            cal.calibrate(
                cal.epsilon_budget(0.001, delta=1e-5), self._process, 0.5, 0.6
            )


# -- Prefix (sequential composition across runs) ------------------------------


class TestCalibratePrefix:
    """Calibrate a second stage over an already-executed prefix."""

    def _stage(self, nm):
        return dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), 0.02) * 200

    def _prefix(self):
        return dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), 0.01) * 100

    def test_matches_manual_closure(self):
        """prefix= is equivalent to composing inside the objective."""
        prefix = self._prefix()
        budget = cal.epsilon_budget(6.0, delta=1e-5)

        via_param = cal.calibrate(budget, self._stage, 0.3, 3.0, prefix=prefix)
        via_closure = cal.calibrate(
            budget, lambda nm: prefix | self._stage(nm), 0.3, 3.0
        )
        assert via_param.param == pytest.approx(via_closure.param)
        assert via_param.achieved == pytest.approx(via_closure.achieved)

    def test_total_hits_budget(self):
        """The composed total (prefix | stage) achieves the target."""
        prefix = self._prefix()
        result = cal.calibrate(
            cal.epsilon_budget(6.0, delta=1e-5), self._stage, 0.3, 3.0, prefix=prefix
        )
        assert result.converged

        total = prefix | self._stage(result.param)
        assert abs(total.epsilon_at(1e-5) - 6.0) < 1e-4

    def test_prefix_demands_more_noise(self):
        """Same budget with a prefix → larger noise multiplier."""
        budget = cal.epsilon_budget(6.0, delta=1e-5)
        without = cal.calibrate(budget, self._stage, 0.3, 3.0)
        with_prefix = cal.calibrate(
            budget, self._stage, 0.3, 3.0, prefix=self._prefix()
        )
        assert with_prefix.param > without.param

    def test_prefix_exhausting_budget_raises(self):
        """Prefix alone above the target → bounds can't bracket."""
        prefix = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.6), 0.02) * 2000
        with pytest.raises(ValueError):
            cal.calibrate(
                cal.epsilon_budget(0.5, delta=1e-5),
                self._stage,
                0.3,
                3.0,
                prefix=prefix,
            )


# -- Calibration roundtrip ---------------------------------------------------


@pytest.mark.slow
class TestCalibrateEpsilon:
    """Calibrate noise multiplier for target epsilon — verify roundtrip."""

    def _process(self, nm):
        return dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), 0.01) * 1000

    def test_basic(self):
        result = cal.calibrate(
            cal.epsilon_budget(5.0, delta=1e-5), self._process, 0.3, 1.2
        )
        assert isinstance(result, CalibrateResult)
        assert result.converged
        assert abs(result.achieved - 5.0) < 1e-4

    def test_strict_epsilon(self):
        result = cal.calibrate(
            cal.epsilon_budget(4.0, delta=1e-5), self._process, 0.3, 1.2
        )
        assert result.converged
        assert abs(result.achieved - 4.0) < 1e-4

    def test_loose_epsilon(self):
        result = cal.calibrate(
            cal.epsilon_budget(8.0, delta=1e-5), self._process, 0.1, 1.0
        )
        assert result.converged
        assert abs(result.achieved - 8.0) < 1e-4

    def test_monotonicity(self):
        """Stricter target → more noise."""
        result_loose = cal.calibrate(
            cal.epsilon_budget(8.0, delta=1e-5), self._process, 0.1, 1.0
        )
        result_strict = cal.calibrate(
            cal.epsilon_budget(4.0, delta=1e-5), self._process, 0.3, 1.2
        )
        # stricter (lower) epsilon requires higher noise_multiplier
        assert result_strict.param > result_loose.param


@pytest.mark.slow
class TestCalibrateDifferentBatchSizes:
    """Calibration converges for various batch/dataset ratios."""

    @pytest.mark.parametrize("batch_size", [8, 32, 128])
    def test_converges(self, batch_size):
        n = 10_000
        q = batch_size / n

        result = cal.calibrate(
            cal.epsilon_budget(5.0, delta=1e-4),
            lambda nm: dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), q) * 1000,
            0.1,
            1.2,
        )
        assert result.converged
        assert abs(result.achieved - 5.0) < 1e-3


@pytest.mark.slow
class TestCalibrateAdvantage:
    """Calibrate for f-DP advantage target."""

    def test_roundtrip(self):
        result = cal.calibrate(
            cal.advantage_budget(0.1),
            lambda nm: dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), 0.01) * 500,
            0.3,
            1.2,
        )
        assert result.converged
        assert abs(result.achieved - 0.1) < 1e-4


@pytest.mark.slow
class TestCalibrateBeta:
    """Calibrate for (α, β) error rate target.

    beta INCREASES with noise_multiplier (more noise → more private → higher
    Type-II error).  The target declares decreasing=False so calibrate()
    handles the direction automatically.
    """

    def test_roundtrip(self):
        result = cal.calibrate(
            cal.beta_budget(0.5, alpha=0.1),
            lambda nm: dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), 0.01) * 500,
            0.3,
            1.2,
        )
        assert result.converged
        assert abs(result.achieved - 0.5) < 1e-3

    def test_monotonicity(self):
        """Stricter (higher) beta target → more noise."""

        def process(nm):
            return dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), 0.01) * 500

        result_low = cal.calibrate(cal.beta_budget(0.3, alpha=0.1), process, 0.3, 1.2)
        result_high = cal.calibrate(cal.beta_budget(0.7, alpha=0.1), process, 0.3, 1.2)
        # Higher beta (more privacy) requires more noise
        assert result_high.param > result_low.param


class TestCalibrateResult:
    """CalibrateResult repr."""

    def test_repr_converged(self):
        r = CalibrateResult(1.0, 3.0, 3.0, 10, True)
        assert "converged" in repr(r)

    def test_repr_not_converged(self):
        r = CalibrateResult(1.0, 3.1, 3.0, 100, False)
        assert "max iterations" in repr(r)
