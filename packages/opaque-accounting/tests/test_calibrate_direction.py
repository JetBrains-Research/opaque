"""Calibration derives the search direction from the calibrated parameter (#333).

Pre-fix, the direction was taken from the budget kind alone (assuming every
parameter behaves like a noise multiplier), so calibrating a parameter whose
increase *spends* privacy (sample rate, step count) raised a
self-contradictory "not in range" error with the target inside the printed
range.

Built entirely on ``acc.eps_delta`` chains: pure opaque-accounting (respects
the test dependency cone) and orders of magnitude faster than PLD mechanisms.
The Poisson/Gaussian integration regressions live in
``packages/opaque-dpsgd/tests/accounting/test_calibration.py``.
"""

from dataclasses import dataclass

import pytest

import opaque.accounting as acc
from opaque.exceptions import CalibrationError

DELTA = 1e-5


@dataclass(frozen=True)
class _MetricProcess:
    metric: float


@dataclass(frozen=True)
class _GainBudget:
    """Synthetic privacy-gain budget (safe at-or-above the target)."""

    value: float
    decreasing: bool = False
    name: str = "gain metric"

    def evaluate(self, process: _MetricProcess) -> float:
        return process.metric


def test_increasing_parameter_loss_metric_converges():
    # Quadrant: privacy-loss metric, epsilon INCREASES with the parameter
    # (step-count-like).  Raised ValueError before the fix.
    result = acc.calibrate(
        acc.epsilon_budget(3.0, delta=DELTA),
        lambda n: acc.eps_delta(0.05, 1e-9) * round(n),
        param_min=10,
        param_max=400,
        tolerance=1e-3,
    )
    assert result.converged
    assert result.achieved <= 3.0
    recheck = acc.epsilon_budget(3.0, delta=DELTA).evaluate(
        acc.eps_delta(0.05, 1e-9) * round(result.param)
    )
    assert recheck == pytest.approx(result.achieved)


def test_decreasing_parameter_loss_metric_converges():
    # Quadrant: privacy-loss metric, epsilon DECREASES with the parameter
    # (noise-multiplier-like) — the previously supported direction.
    result = acc.calibrate(
        acc.epsilon_budget(3.0, delta=DELTA),
        lambda nm: acc.eps_delta(1.0 / nm, 1e-9) * 100,
        param_min=0.5,
        param_max=64.0,
        tolerance=1e-4,
    )
    assert result.converged
    assert result.achieved <= 3.0


@pytest.mark.parametrize(
    ("metric", "quadrant"),
    [
        (lambda p: 1.0 - 0.1 * p, "gain-decreasing (safe end = param_min)"),
        (lambda p: 0.1 * p, "gain-increasing (safe end = param_max)"),
    ],
)
def test_gain_metric_quadrants_converge(metric, quadrant):
    # Quadrants 3+4: privacy-gain metric (safe at-or-above the target), for
    # both parameter directions.  Beta/risk are degenerate on pure
    # ``eps_delta`` chains, so a synthetic gain budget pins the direction
    # logic exactly (the logic is metric-agnostic); real-mechanism beta
    # calibrations live in the dpsgd suite.
    result = acc.calibrate(
        _GainBudget(0.5),
        lambda p: _MetricProcess(metric(p)),
        param_min=0.0,
        param_max=8.0,
        tolerance=1e-6,
    )
    assert result.converged, quadrant
    assert result.achieved >= 0.5, quadrant


def test_inf_on_unsafe_endpoint_accepted():
    # epsilon is continuous over most of the bracket but infinite at the
    # unsafe endpoint (the process's delta exceeds the demanded target
    # there): inf on the UNSAFE end must be accepted — the target is below
    # it by definition — and calibration converges on the finite side.
    def process(x):
        return acc.eps_delta(x, 1e-3 if x > 5.0 else 1e-9)

    result = acc.calibrate(
        acc.epsilon_budget(3.0, delta=1e-6),
        process,
        param_min=0.5,
        param_max=8.0,
        tolerance=1e-4,
    )
    assert result.converged
    assert result.achieved <= 3.0


def test_flat_parameterization_rejected():
    with pytest.raises(CalibrationError, match="flat"):
        acc.calibrate(
            acc.epsilon_budget(3.0, delta=DELTA),
            lambda _p: acc.eps_delta(0.05, 1e-9) * 10,
            param_min=0.1,
            param_max=5.0,
        )


def test_both_endpoints_safe_rejected():
    # V-shaped epsilon with BOTH endpoints privacy-safe: caught at the
    # bracket check before any bisection.
    with pytest.raises(CalibrationError, match="not bracketed"):
        acc.calibrate(
            acc.epsilon_budget(4.0, delta=DELTA),
            lambda x: acc.eps_delta(0.4 + abs(x - 1.0) * 3.0, 1e-9),
            param_min=0.0,
            param_max=1.9,
            tolerance=1e-4,
        )


def test_non_monotone_interior_probe_rejected():
    # V-shaped epsilon straddling the target (param_min unsafe, param_max
    # safe): the first interior probe dips below the endpoint value
    # envelope and must raise "not monotone" instead of mis-calibrating.
    with pytest.raises(CalibrationError, match="not monotone"):
        acc.calibrate(
            acc.epsilon_budget(2.0, delta=DELTA),
            lambda x: acc.eps_delta(0.4 + abs(x - 1.0) * 3.0, 1e-9),
            param_min=0.0,
            param_max=1.4,
            tolerance=1e-4,
        )
