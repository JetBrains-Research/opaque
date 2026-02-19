"""Calibration utilities for finding noise multipliers that achieve target privacy.

This module provides a binary search framework for calibrating noise multipliers
to achieve specific privacy targets:

- **epsilon_budget(...)**: Calibrate for (ε, δ)-DP
- **delta_budget(...)**: Calibrate for (ε, δ)-DP (inverse direction)
- **advantage_budget(...)**: Calibrate for f-DP advantage
- **beta_budget(...)**: Calibrate for (α, β) error rates
- **risk_budget(...)**: Calibrate for Bayes risk

Example::

    from opaque.accounting import calibration as cal
    import opaque.accounting as acc

    # Find noise multiplier for target privacy budget
    target = cal.epsilon_budget(3.0, delta=1e-5)
    result = cal.calibrate(
        target,
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
from typing import Protocol

from opaque.accounting import DpProcess


# =============================================================================
# Target protocol and implementations
# =============================================================================


class Target(Protocol):
    """Protocol for calibration targets.

    A target defines:
    - **evaluate(process)**: Compute metric value for a process
    - **value**: Target value to achieve
    - **name**: Human-readable name for debugging
    - **decreasing**: Whether the metric decreases as the calibrated parameter
      increases.  For noise_multiplier calibration this is ``True`` for
      privacy-loss metrics (epsilon, delta, advantage) and ``False`` for
      privacy-gain metrics (beta, risk) which *increase* with noise.
    """

    value: float
    name: str
    decreasing: bool

    def evaluate(self, process: DpProcess) -> float:
        """Evaluate the metric on a DP process.

        Args:
            process: The DP process to evaluate.

        Returns:
            Metric value (e.g., epsilon, advantage, beta).
        """
        ...


@dataclass
class EpsilonTarget:
    """Target for (ε, δ)-DP: find noise achieving target epsilon at given delta.
    
    Parameters must satisfy: epsilon > 0 and delta in (0, 1).
    """

    epsilon: float
    delta: float

    def __post_init__(self) -> None:
        """Validate target parameters."""
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if not (0 < self.delta < 1):
            raise ValueError(f"delta must be in (0, 1), got {self.delta}")

    @property
    def value(self) -> float:
        return self.epsilon

    @property
    def name(self) -> str:
        return f"epsilon({self.epsilon}, delta={self.delta})"

    @property
    def decreasing(self) -> bool:
        return True

    def evaluate(self, process: DpProcess) -> float:
        """Get epsilon at the target delta."""
        return process.epsilon_at(self.delta)


@dataclass
class DeltaTarget:
    """Target for (ε, δ)-DP: find noise achieving target delta at given epsilon.
    
    Parameters must satisfy: delta in (0, 1) and epsilon > 0.
    """

    delta: float
    epsilon: float

    def __post_init__(self) -> None:
        """Validate target parameters."""
        if not (0 < self.delta < 1):
            raise ValueError(f"delta must be in (0, 1), got {self.delta}")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")

    @property
    def value(self) -> float:
        return self.delta

    @property
    def name(self) -> str:
        return f"delta({self.delta}, epsilon={self.epsilon})"

    @property
    def decreasing(self) -> bool:
        return True

    def evaluate(self, process: DpProcess) -> float:
        """Get delta at the target epsilon."""
        return process.delta_at(self.epsilon)


@dataclass
class AdvantageTarget:
    """Target for f-DP: find noise achieving target advantage.
    
    Advantage must be in (0, 1) — represents total-variation distance between
    neighboring dataset distributions.
    """

    advantage: float

    def __post_init__(self) -> None:
        """Validate target parameter."""
        if not (0 < self.advantage < 1):
            raise ValueError(f"advantage must be in (0, 1), got {self.advantage}")

    @property
    def value(self) -> float:
        return self.advantage

    @property
    def name(self) -> str:
        return f"advantage({self.advantage})"

    @property
    def decreasing(self) -> bool:
        return True

    def evaluate(self, process: DpProcess) -> float:
        """Get f-DP advantage."""
        return process.advantage()


@dataclass
class BetaTarget:
    """Target for (α, β) error rates: find noise achieving target beta at given alpha.
    
    Parameters must satisfy: 0 < alpha < 1 and 0 < beta < 1.
    """

    beta: float
    alpha: float

    def __post_init__(self) -> None:
        """Validate target parameters."""
        if not (0 < self.beta < 1):
            raise ValueError(f"beta must be in (0, 1), got {self.beta}")
        if not (0 < self.alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")

    @property
    def value(self) -> float:
        return self.beta

    @property
    def name(self) -> str:
        return f"beta({self.beta}, alpha={self.alpha})"

    @property
    def decreasing(self) -> bool:
        return False

    def evaluate(self, process: DpProcess) -> float:
        """Get beta at the target alpha."""
        return process.beta_at(self.alpha)


@dataclass
class RiskTarget:
    """Target for Bayes risk: find noise achieving target risk at given prior.
    
    Parameters must satisfy: risk in (0, 1) and prior in (0, 1).
    """

    risk: float
    prior: float

    def __post_init__(self) -> None:
        """Validate target parameters."""
        if not (0 < self.risk < 1):
            raise ValueError(f"risk must be in (0, 1), got {self.risk}")
        if not (0 < self.prior < 1):
            raise ValueError(f"prior must be in (0, 1), got {self.prior}")

    @property
    def value(self) -> float:
        return self.risk

    @property
    def name(self) -> str:
        return f"risk({self.risk}, prior={self.prior})"

    @property
    def decreasing(self) -> bool:
        return False

    def evaluate(self, process: DpProcess) -> float:
        """Get Bayes risk at the target prior."""
        return process.risk_at(self.prior)


# =============================================================================
# Target factories
# =============================================================================


def epsilon_budget(epsilon: float, delta: float) -> EpsilonTarget:
    """Create a target for (ε, δ)-DP: find noise achieving target epsilon.

    Args:
        epsilon: Target ε value.
        delta: Fixed δ value.

    Returns:
        Calibration target.

    Example::

        target = cal.epsilon_budget(3.0, delta=1e-5)
        result = cal.calibrate(
            target,
            lambda nm: acc.poisson(acc.gaussian(nm), 0.01) * 1000,
            0.1, 5.0,
        )
    """
    return EpsilonTarget(epsilon=epsilon, delta=delta)


def delta_budget(delta: float, epsilon: float) -> DeltaTarget:
    """Create a target for (ε, δ)-DP: find noise achieving target delta.

    Args:
        delta: Target δ value.
        epsilon: Fixed ε value.

    Returns:
        Calibration target.

    Example::

        target = cal.delta_budget(1e-6, epsilon=3.0)
        result = cal.calibrate(
            target,
            lambda nm: acc.poisson(acc.gaussian(nm), 0.01) * 1000,
            0.1, 5.0,
        )
    """
    return DeltaTarget(delta=delta, epsilon=epsilon)


def advantage_budget(advantage: float) -> AdvantageTarget:
    """Create a target for f-DP: find noise achieving target advantage.

    Args:
        advantage: Target advantage value (TV distance between neighboring datasets).

    Returns:
        Calibration target.

    Example::

        target = cal.advantage_budget(0.1)
        result = cal.calibrate(
            target,
            lambda nm: acc.poisson(acc.gaussian(nm), 0.01) * 1000,
            0.1, 5.0,
        )
    """
    return AdvantageTarget(advantage=advantage)


def beta_budget(beta: float, alpha: float) -> BetaTarget:
    """Create a target for (α, β) error rates: find noise achieving target beta.

    Args:
        beta: Target Type-II error rate.
        alpha: Fixed Type-I error rate.

    Returns:
        Calibration target.

    Example::

        target = cal.beta_budget(0.05, alpha=0.01)
        result = cal.calibrate(
            target,
            lambda nm: acc.poisson(acc.gaussian(nm), 0.01) * 1000,
            0.1, 5.0,
        )
    """
    return BetaTarget(beta=beta, alpha=alpha)


def risk_budget(risk: float, prior: float) -> RiskTarget:
    """Create a target for Bayes risk: find noise achieving target risk.

    Args:
        risk: Target Bayes risk.
        prior: Prior probability of hypothesis H1.

    Returns:
        Calibration target.

    Example::

        target = cal.risk_budget(0.1, prior=0.5)
        result = cal.calibrate(
            target,
            lambda nm: acc.poisson(acc.gaussian(nm), 0.01) * 1000,
            0.1, 5.0,
        )
    """
    return RiskTarget(risk=risk, prior=prior)


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
    target: Target,
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
    Each target declares a ``decreasing`` property that tells the search
    whether the metric decreases (True) or increases (False) as the
    calibrated parameter grows.  Epsilon/delta/advantage are decreasing;
    beta/risk are increasing.  The binary search adapts automatically.

    **Parameters:**

    Args:
        target: Calibration target (created with epsilon(), delta(), etc.)
            - Must have: target.value (float), target.evaluate(process) → float
            - Common targets: epsilon(3.0, 1e-5), delta(1e-5, 3.0), advantage(0.1)
            - Targets validate themselves: epsilon(-1.0) raises ValueError

        process: Callable taking a float parameter and returning a DpProcess.
            Must be deterministic (same input → same process).
            Example: ``lambda nm: acc.poisson(acc.gaussian(nm), 0.01) * 1000``
            
            Important: If process() raises an exception, it propagates immediately.

        param_min: Lower bound for search (usually produces more private result)
            - Assumed to satisfy: metric(param_min) > target.value
            - Example: 0.5 for noise_multiplier at high privacy

        param_max: Upper bound for search (usually produces less private result)
            - Assumed to satisfy: metric(param_max) < target.value
            - Example: 3.0 for noise_multiplier at lower privacy

        tolerance: Convergence threshold
            - Stops early when |achieved - target.value| < tolerance
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
        ValueError: If target evaluation returns inf or nan at the bounds
        Exception: If process() or target.evaluate() raises an exception

    **Examples:**

    **Example 1: Standard (ε, δ)-DP**::

        import opaque.accounting as acc
        from opaque.accounting import calibration as cal

        target = cal.epsilon_budget(3.0, delta=1e-5)
        result = cal.calibrate(
            target,
            lambda nm: acc.poisson(acc.gaussian(nm), sample_rate=0.01) * 1000,
            param_min=0.7,
            param_max=1.2,
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

    - *ValueError: bounds don't bracket target*
      → Try expanding param_min/param_max range
    - *ValueError: target evaluation returned infinity*
      → Privacy target may be impossible with your mechanism; try higher param_min or lower target
    - *Not converging within max_iterations*
      → Increase tolerance or max_iterations; check that param changes actually affect metric
    """
    if param_min >= param_max:
        raise ValueError(
            f"param_min ({param_min}) must be < param_max ({param_max})"
        )

    # Check bounds bracket the target
    proc_min = process(param_min)
    proc_max = process(param_max)
    val_min = target.evaluate(proc_min)
    val_max = target.evaluate(proc_max)

    # Validate bounds don't return inf/nan
    if math.isnan(val_min) or math.isnan(val_max):
        raise ValueError(
            f"Target evaluation returned NaN at bounds: "
            f"at param_min={param_min}: {val_min}, at param_max={param_max}: {val_max}. "
            f"This usually indicates an issue with the process() function."
        )

    if math.isinf(val_min) or math.isinf(val_max):
        raise ValueError(
            f"Target evaluation returned infinity at bounds: "
            f"at param_min={param_min}: {val_min}, at param_max={param_max}: {val_max}. "
            f"This typically means the privacy target is unreachable with these parameter bounds. "
            f"Try expanding the search range or checking that process() produces valid DpProcess objects."
        )

    # Determine search direction from the target
    decreasing = target.decreasing
    if decreasing:
        # metric decreases with param: val_min is high, val_max is low
        lo_val, hi_val = val_min, val_max
    else:
        # metric increases with param: val_min is low, val_max is high
        lo_val, hi_val = val_max, val_min

    if not (hi_val <= target.value <= lo_val):
        raise ValueError(
            f"Target {target.name}={target.value:.6f} not in range "
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
        proc = process(mid)
        current = target.evaluate(proc)

        # Check convergence
        if abs(current - target.value) < tolerance:
            return CalibrateResult(
                param=mid,
                achieved=current,
                target=target.value,
                iterations=iterations,
                converged=True,
            )

        # Update bounds using target's declared direction
        if (current > target.value) == decreasing:
            lo = mid
        else:
            hi = mid

    # Max iterations reached - return best estimate
    mid = (lo + hi) / 2
    proc = process(mid)
    current = target.evaluate(proc)

    return CalibrateResult(
        param=mid,
        achieved=current,
        target=target.value,
        iterations=iterations,
        converged=False,
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Targets
    "Target",
    "EpsilonTarget",
    "DeltaTarget",
    "AdvantageTarget",
    "BetaTarget",
    "RiskTarget",
    # Target factories
    "epsilon_budget",
    "delta_budget",
    "advantage_budget",
    "beta_budget",
    "risk_budget",
    # Calibration
    "calibrate",
    "CalibrateResult",
]
