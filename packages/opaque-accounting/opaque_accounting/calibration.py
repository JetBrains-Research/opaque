"""Binary search calibration for finding parameters that achieve target privacy.

This module provides a generic binary search framework.  Privacy budget
*budgets* (epsilon, delta, advantage, beta, risk) live in
:mod:`opaque.accounting.budgets` and are re-exported here for convenience.

Example::

    from opaque_accounting import calibration as cal
    import opaque_accounting as acc

    budget = cal.epsilon_budget(3.0, delta=1e-5)
    result = cal.calibrate(
        budget,
        lambda nm: (acc.poisson(acc.gaussian(nm), sample_rate=0.01) * 1000).cgf(),
        param_min=0.1, param_max=5.0,
    )

    print(f"Noise multiplier: {result.param:.3f}")
    print(f"Achieved epsilon: {result.achieved:.6f}")
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from opaque_accounting.base import Pld

# Re-export budgets so ``from opaque_accounting.calibration import Budget``
# and ``from opaque_accounting import calibration as cal; cal.epsilon_budget(...)``
# work as convenience imports.
from opaque_accounting.budgets import (  # noqa: F401
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
    process: Callable[[float], Pld],
    param_min: float,
    param_max: float,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> CalibrateResult:
    """Binary search for parameter achieving target privacy metric.

    Finds the value of a parameter (e.g., noise_multiplier) such that
    the resulting PLD achieves a target privacy guarantee.

    Args:
        budget: Calibration budget (created with epsilon_budget(), delta_budget(), etc.)
            - Must have: budget.value (float), budget.evaluate(pld) → float

        process: Callable taking a float parameter and returning a materialized Pld.
            Must be deterministic (same input → same output).
            Example: ``lambda nm: (acc.poisson(acc.gaussian(nm), 0.01) * 1000).cgf()``

        param_min: Lower bound for search.
        param_max: Upper bound for search.
        tolerance: Convergence threshold (default: 1e-6).
        max_iterations: Maximum binary search iterations (default: 100).

    Returns:
        CalibrateResult with param, achieved, target, iterations, converged.

    Raises:
        ValueError: If param_min >= param_max, bounds don't bracket, or NaN/inf.

    Example::

        budget = cal.epsilon_budget(3.0, delta=1e-5)
        result = cal.calibrate(
            budget,
            lambda nm: (acc.poisson(acc.gaussian(nm), 0.01) * 1000).cgf(),
            param_min=0.7, param_max=1.2,
        )
        print(f"Use noise_multiplier = {result.param:.4f}")
    """
    if param_min >= param_max:
        raise ValueError(f"param_min ({param_min}) must be < param_max ({param_max})")

    # Check bounds bracket the budget
    pld_min = process(param_min)
    pld_max = process(param_max)
    val_min = budget.evaluate(pld_min)
    val_max = budget.evaluate(pld_max)

    # Validate bounds don't return inf/nan
    if math.isnan(val_min) or math.isnan(val_max):
        raise ValueError(
            f"Budget evaluation returned NaN at bounds: "
            f"at param_min={param_min}: {val_min}, at param_max={param_max}: {val_max}. "
            f"This usually indicates an issue with the process() function."
        )

    if math.isinf(val_min) or math.isinf(val_max):
        raise ValueError(
            f"Budget evaluation returned infinity at bounds: "
            f"at param_min={param_min}: {val_min}, at param_max={param_max}: {val_max}. "
            f"This typically means the privacy target is unreachable with these parameter bounds. "
            f"Try expanding the search range."
        )

    # Determine search direction from the budget
    decreasing = budget.decreasing
    if decreasing:
        lo_val, hi_val = val_min, val_max
    else:
        lo_val, hi_val = val_max, val_min

    if not (hi_val <= budget.value <= lo_val):
        raise ValueError(
            f"Budget {budget.name}={budget.value:.6f} not in range "
            f"[{min(val_min, val_max):.6f}, {max(val_min, val_max):.6f}] "
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
        pld = process(mid)
        current = budget.evaluate(pld)

        if math.isnan(current):
            raise ValueError(
                f"Budget evaluation returned NaN at param={mid} "
                f"(iteration {iterations}). Check that process() "
                f"produces valid Pld objects for all parameter values."
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
    pld = process(mid)
    current = budget.evaluate(pld)

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
