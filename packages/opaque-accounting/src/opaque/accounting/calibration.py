"""Binary search calibration for finding parameters that achieve target privacy.

This module provides a generic binary search framework.  Privacy budget
*budgets* (epsilon, delta, advantage, beta, risk) live in
:mod:`opaque.accounting.budgets` and are re-exported here for convenience.

Example::

    from opaque.accounting import calibration as cal
    import opaque.accounting as acc

    budget = cal.epsilon_budget(3.0, delta=1e-5)
    result = cal.calibrate(
        budget,
        lambda nm: acc.poisson(acc.gaussian(nm), sample_rate=0.01) * 1000,
        param_min=0.1, param_max=5.0,
    )

    print(f"Noise multiplier: {result.param:.3f}")
    print(f"Achieved epsilon: {result.achieved:.6f}")
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from opaque.accounting.base import DpProcess

# Re-export budgets so ``from opaque.accounting.calibration import Budget``
# and ``from opaque.accounting import calibration as cal; cal.epsilon_budget(...)``
# work as convenience imports.
from opaque.accounting.budgets import (  # noqa: F401
    AdvantageBudget,
    BetaBudget,
    Budget,
    DeltaBudget,
    EpsilonBudget,
    RiskBudget,
    advantage_budget,
    beta_budget,
    delta_budget,
    epsilon_budget,
    risk_budget,
)

# =============================================================================
# Calibration
# =============================================================================


@dataclass
class CalibrateResult:
    """Result from calibration binary search.

    Attributes:
        param: Found parameter value (e.g., noise_multiplier).
        achieved: Achieved metric value.
        target: Target metric value.
        iterations: Number of binary search iterations.
        converged: Whether calibration converged within tolerance.
    """

    param: float
    achieved: float
    target: float
    iterations: int
    converged: bool

    def __repr__(self) -> str:
        status = "converged" if self.converged else "max iterations"
        return (
            f"CalibrateResult(param={self.param:.6f}, "
            f"achieved={self.achieved:.6f}, target={self.target:.6f}, "
            f"iterations={self.iterations}, {status})"
        )


def calibrate(
    budget: Budget,
    process: Callable[[float], DpProcess],
    param_min: float,
    param_max: float,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> CalibrateResult:
    """Binary search for parameter achieving target privacy metric.

    Finds the value of a parameter (e.g., noise_multiplier) such that
    the resulting DpProcess achieves a target privacy guarantee.

    **Metric Direction:**
    Each budget declares a ``decreasing`` property that tells the search
    whether the metric decreases (True) or increases (False) as the
    calibrated parameter grows.  Epsilon/delta/advantage are decreasing;
    beta/risk are increasing.  The binary search adapts automatically.

    **Parameters:**

    Args:
        budget: Calibration budget (created with epsilon_budget(), delta_budget(), etc.)
            - Must have: budget.value (float), budget.evaluate(process) → float
            - Common budgets: epsilon_budget(3.0, 1e-5), delta_budget(1e-5, 3.0), advantage_budget(0.1)
            - Budgets validate themselves: epsilon_budget(-1.0) raises ValueError

        process: Callable taking a float parameter and returning a DpProcess.
            Must be deterministic (same input → same process).
            Example: ``lambda nm: acc.poisson(acc.gaussian(nm), 0.01) * 1000``

            Important: If process() raises an exception, it propagates immediately.

        param_min: Lower bound for search (usually produces more private result)
            - Assumed to satisfy: metric(param_min) > budget.value
            - Example: 0.5 for noise_multiplier at high privacy

        param_max: Upper bound for search (usually produces less private result)
            - Assumed to satisfy: metric(param_max) < budget.value
            - Example: 3.0 for noise_multiplier at lower privacy

        tolerance: Convergence threshold
            - Stops early when |achieved - budget.value| < tolerance
            - Default: 1e-6 (very tight, suitable for most applications)
            - Use 1e-2 for faster convergence, 1e-8 for maximum precision

        max_iterations: Maximum binary search iterations
            - Each iteration halves the search space
            - 100 iterations gives ~1e-30 precision (rarely needed)
            - If not converged after max_iterations, returns False for converged

    Returns:
        CalibrateResult with:
        - param: Found parameter value
        - achieved: Metric value at found parameter
        - target: Target metric value (for comparison)
        - iterations: Number of iterations performed
        - converged: True if |achieved - target| < tolerance

    Raises:
        ValueError: If param_min >= param_max
        ValueError: If bounds don't bracket the target (both val_min/val_max above or below target)
        ValueError: If budget evaluation returns inf or nan at the bounds
        Exception: If process() or budget.evaluate() raises an exception

    **Examples:**

    **Example 1: Standard (ε, δ)-DP**::

        import opaque.accounting as acc
        from opaque.accounting import calibration as cal

        budget = cal.epsilon_budget(3.0, delta=1e-5)
        result = cal.calibrate(
            budget,
            lambda nm: acc.poisson(acc.gaussian(nm), sample_rate=0.01) * 1000,
            param_min=0.7,
            param_max=3.5,
        )

        print(f"Use noise_multiplier = {result.param:.4f}")
        print(f"Achieves epsilon = {result.achieved:.6f}")
        print(f"Converged: {result.converged}")

    **Example 2: Multi-phase training**::

        def multiphase(nm):
            phase1 = acc.poisson(acc.gaussian(nm), 0.01) * 500
            phase2 = acc.poisson(acc.gaussian(nm * 0.8), 0.01) * 500
            phase3 = acc.poisson(acc.gaussian(nm * 0.5), 0.01) * 500
            return phase1 | phase2 | phase3

        result = cal.calibrate(
            cal.epsilon_budget(5.0, delta=1e-5),
            multiphase,
            param_min=0.5,
            param_max=3.0,
            tolerance=0.01,
        )

    **Example 3: Different privacy metrics**::

        process = lambda nm: acc.poisson(acc.gaussian(nm), 0.01) * 1000

        # f-DP advantage
        result = cal.calibrate(cal.advantage_budget(0.1), process, 0.5, 2.0)

        # (α, β) error rates
        result = cal.calibrate(cal.beta_budget(0.05, alpha=0.01), process, 0.5, 2.0)

        # Bayes risk
        result = cal.calibrate(cal.risk_budget(0.1, prior=0.5), process, 0.5, 2.0)

    **Troubleshooting:**

    - *ValueError: bounds don't bracket budget*
      → Try expanding param_min/param_max range
    - *ValueError: budget evaluation returned infinity*
      → Privacy budget may be unreachable with your mechanism; try higher param_min or lower budget value
    - *Not converging within max_iterations*
      → Increase tolerance or max_iterations; check that param changes actually affect metric
    """
    if param_min >= param_max:
        raise ValueError(f"param_min ({param_min}) must be < param_max ({param_max})")

    # Check bounds bracket the budget
    proc_min = process(param_min)
    proc_max = process(param_max)
    val_min = budget.evaluate(proc_min)
    val_max = budget.evaluate(proc_max)

    # Validate bounds don't return nan
    if math.isnan(val_min) or math.isnan(val_max):
        raise ValueError(
            f"Budget evaluation returned NaN at bounds: "
            f"at param_min={param_min}: {val_min}, at param_max={param_max}: {val_max}. "
            f"This usually indicates an issue with the process() function."
        )

    if math.isinf(val_min) and math.isinf(val_max):
        raise ValueError(
            f"Budget evaluation returned infinity at both bounds: "
            f"at param_min={param_min}: {val_min}, at param_max={param_max}: {val_max}. "
            f"This typically means the privacy target is unreachable with these parameter bounds. "
            f"Try expanding the search range or checking that process() produces valid DpProcess objects."
        )

    # Determine search direction from the budget
    decreasing = budget.decreasing
    if decreasing:
        # metric decreases with param: val_min is high, val_max is low
        lo_val, hi_val = val_min, val_max
    else:
        # metric increases with param: val_min is low, val_max is high
        lo_val, hi_val = val_max, val_min

    # Bracketing check: inf on the "high" side is fine (target < inf),
    # but inf on the "low" side means target is unreachable.
    if math.isinf(hi_val):
        raise ValueError(
            f"Budget evaluation returned infinity on the wrong bound: "
            f"at param_min={param_min}: {val_min}, at param_max={param_max}: {val_max}. "
            f"This typically means the privacy target is unreachable with these parameter bounds. "
            f"Try expanding the search range or checking that process() produces valid DpProcess objects."
        )

    if not math.isinf(lo_val) and not (hi_val <= budget.value <= lo_val):
        raise ValueError(
            f"Budget {budget.name}={budget.value:.6f} not in range "
            f"[{min(val_min, val_max):.6f}, {max(val_min, val_max):.6f}] "
            f"for param range [{param_min}, {param_max}]. "
            f"The target may be unreachable with these bounds."
        )

    if math.isinf(lo_val) and budget.value < hi_val:
        raise ValueError(
            f"Budget {budget.name}={budget.value:.6f} below finite bound {hi_val:.6f} "
            f"for param range [{param_min}, {param_max}]. "
            f"The target may be unreachable with these bounds."
        )

    # Binary search
    lo = param_min
    hi = param_max
    iterations = 0

    for iteration in range(max_iterations):
        iterations = iteration + 1
        mid = (lo + hi) / 2
        proc = process(mid)
        current = budget.evaluate(proc)

        if math.isnan(current):
            raise ValueError(
                f"Budget evaluation returned NaN at param={mid} "
                f"(iteration {iterations}). Check that process() "
                f"produces valid DpProcess objects for all parameter values."
            )

        # Check convergence
        if abs(current - budget.value) < tolerance:
            return CalibrateResult(
                param=mid,
                achieved=current,
                target=budget.value,
                iterations=iterations,
                converged=True,
            )

        # Update bounds using budget's declared direction
        if (current > budget.value) == decreasing:
            lo = mid
        else:
            hi = mid

    # Max iterations reached - return best estimate
    mid = (lo + hi) / 2
    proc = process(mid)
    current = budget.evaluate(proc)

    return CalibrateResult(
        param=mid,
        achieved=current,
        target=budget.value,
        iterations=iterations,
        converged=False,
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Re-exported from budgets (convenience)
    "Budget",
    "EpsilonBudget",
    "DeltaBudget",
    "AdvantageBudget",
    "BetaBudget",
    "RiskBudget",
    "epsilon_budget",
    "delta_budget",
    "advantage_budget",
    "beta_budget",
    "risk_budget",
    # Calibration
    "calibrate",
    "CalibrateResult",
]
