"""Calibration utilities for finding noise multipliers that achieve target privacy.

This module provides a binary search framework for calibrating noise multipliers
to achieve specific privacy targets:

- **epsilon(...)**: Calibrate for (ε, δ)-DP
- **delta(...)**: Calibrate for (ε, δ)-DP (inverse direction)
- **advantage(...)**: Calibrate for f-DP advantage
- **beta(...)**: Calibrate for (α, β) error rates
- **risk(...)**: Calibrate for Bayes risk

Example::

    from opaque.accounting import calibration as cal
    import opaque.accounting as acc

    # Find noise multiplier for target privacy budget
    def build(nm):
        return acc.poisson(nm, sample_rate=0.01) * 1000

    target = cal.epsilon(3.0, delta=1e-5)
    result = cal.calibrate(target, build, param_min=0.1, param_max=5.0)

    print(f"Noise multiplier: {result.param:.3f}")
    print(f"Achieved epsilon: {result.achieved:.6f}")
"""

from dataclasses import dataclass
from typing import Callable, Protocol

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
    """

    value: float
    name: str

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
    """Target for (ε, δ)-DP: find noise achieving target epsilon at given delta."""

    epsilon: float
    delta: float

    @property
    def value(self) -> float:
        return self.epsilon

    @property
    def name(self) -> str:
        return f"epsilon({self.epsilon}, delta={self.delta})"

    def evaluate(self, process: DpProcess) -> float:
        """Get epsilon at the target delta."""
        return process.epsilon_at(self.delta)


@dataclass
class DeltaTarget:
    """Target for (ε, δ)-DP: find noise achieving target delta at given epsilon."""

    delta: float
    epsilon: float

    @property
    def value(self) -> float:
        return self.delta

    @property
    def name(self) -> str:
        return f"delta({self.delta}, epsilon={self.epsilon})"

    def evaluate(self, process: DpProcess) -> float:
        """Get delta at the target epsilon."""
        return process.delta_at(self.epsilon)


@dataclass
class AdvantageTarget:
    """Target for f-DP: find noise achieving target advantage."""

    advantage: float

    @property
    def value(self) -> float:
        return self.advantage

    @property
    def name(self) -> str:
        return f"advantage({self.advantage})"

    def evaluate(self, process: DpProcess) -> float:
        """Get f-DP advantage."""
        return process.advantage()


@dataclass
class BetaTarget:
    """Target for (α, β) error rates: find noise achieving target beta at given alpha."""

    beta: float
    alpha: float

    @property
    def value(self) -> float:
        return self.beta

    @property
    def name(self) -> str:
        return f"beta({self.beta}, alpha={self.alpha})"

    def evaluate(self, process: DpProcess) -> float:
        """Get beta at the target alpha."""
        return process.beta_at(self.alpha)


@dataclass
class RiskTarget:
    """Target for Bayes risk: find noise achieving target risk at given prior."""

    risk: float
    prior: float

    @property
    def value(self) -> float:
        return self.risk

    @property
    def name(self) -> str:
        return f"risk({self.risk}, prior={self.prior})"

    def evaluate(self, process: DpProcess) -> float:
        """Get Bayes risk at the target prior."""
        return process.risk_at(self.prior)


# =============================================================================
# Target factories
# =============================================================================


def epsilon(epsilon: float, delta: float) -> EpsilonTarget:
    """Create a target for (ε, δ)-DP: find noise achieving target epsilon.

    Args:
        epsilon: Target ε value.
        delta: Fixed δ value.

    Returns:
        Calibration target.

    Example::

        target = cal.epsilon(3.0, delta=1e-5)
        result = cal.calibrate(target, build_fn, 0.1, 5.0)
    """
    return EpsilonTarget(epsilon=epsilon, delta=delta)


def delta(delta: float, epsilon: float) -> DeltaTarget:
    """Create a target for (ε, δ)-DP: find noise achieving target delta.

    Args:
        delta: Target δ value.
        epsilon: Fixed ε value.

    Returns:
        Calibration target.

    Example::

        target = cal.delta(1e-6, epsilon=3.0)
        result = cal.calibrate(target, build_fn, 0.1, 5.0)
    """
    return DeltaTarget(delta=delta, epsilon=epsilon)


def advantage(advantage: float) -> AdvantageTarget:
    """Create a target for f-DP: find noise achieving target advantage.

    Args:
        advantage: Target advantage value (TV distance between neighboring datasets).

    Returns:
        Calibration target.

    Example::

        target = cal.advantage(0.1)
        result = cal.calibrate(target, build_fn, 0.1, 5.0)
    """
    return AdvantageTarget(advantage=advantage)


def beta(beta: float, alpha: float) -> BetaTarget:
    """Create a target for (α, β) error rates: find noise achieving target beta.

    Args:
        beta: Target Type-II error rate.
        alpha: Fixed Type-I error rate.

    Returns:
        Calibration target.

    Example::

        target = cal.beta(0.05, alpha=0.01)
        result = cal.calibrate(target, build_fn, 0.1, 5.0)
    """
    return BetaTarget(beta=beta, alpha=alpha)


def risk(risk: float, prior: float) -> RiskTarget:
    """Create a target for Bayes risk: find noise achieving target risk.

    Args:
        risk: Target Bayes risk.
        prior: Prior probability of hypothesis H1.

    Returns:
        Calibration target.

    Example::

        target = cal.risk(0.1, prior=0.5)
        result = cal.calibrate(target, build_fn, 0.1, 5.0)
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
    build: Callable[[float], DpProcess],
    param_min: float,
    param_max: float,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> CalibrateResult:
    """Binary search for parameter achieving target privacy metric.

    Searches for ``param`` in ``[param_min, param_max]`` such that:
        ``target.evaluate(build(param)) ≈ target.value``

    The search assumes:
    - Metric **decreases** as param increases (e.g., epsilon vs noise_multiplier)
    - If your metric increases, swap param_min/param_max or invert the metric

    Args:
        target: Calibration target (created with epsilon(), beta(), etc.).
        build: Function taking parameter value and returning a DpProcess.
            Example: ``lambda nm: acc.poisson(nm, 0.01) * 1000``
        param_min: Lower bound for parameter search.
        param_max: Upper bound for parameter search.
        tolerance: Stop when ``|achieved - target| < tolerance``. Default: 1e-6.
        max_iterations: Maximum number of binary search iterations. Default: 100.

    Returns:
        CalibrateResult with found parameter and achieved value.

    Raises:
        ValueError: If param_min >= param_max or if bounds bracket the target incorrectly.

    Example::

        # Find noise multiplier for (ε=3.0, δ=1e-5)
        def build(nm):
            return acc.poisson(nm, sample_rate=0.01) * 1000

        target = cal.epsilon(3.0, delta=1e-5)
        result = cal.calibrate(target, build, param_min=0.1, param_max=5.0)

        print(f"Use noise_multiplier = {result.param:.3f}")
        print(f"Achieves epsilon = {result.achieved:.6f}")

        # Multi-phase training
        def build_multiphase(nm):
            phase1 = acc.poisson(nm, 0.01) * 500
            phase2 = acc.poisson(nm * 0.8, 0.01) * 500
            return phase1 | phase2

        result = cal.calibrate(
            cal.epsilon(5.0, delta=1e-5),
            build_multiphase,
            param_min=0.5,
            param_max=3.0,
        )
    """
    if param_min >= param_max:
        raise ValueError(
            f"param_min ({param_min}) must be < param_max ({param_max})"
        )

    # Check bounds bracket the target
    proc_min = build(param_min)
    proc_max = build(param_max)
    val_min = target.evaluate(proc_min)
    val_max = target.evaluate(proc_max)

    # We assume metric decreases as param increases
    # (e.g., epsilon decreases as noise_multiplier increases)
    if not (val_max <= target.value <= val_min):
        raise ValueError(
            f"Target {target.name}={target.value:.6f} not in range "
            f"[{val_max:.6f}, {val_min:.6f}] for param range [{param_min}, {param_max}]. "
            f"The target may be unreachable with these bounds, or the metric may "
            f"increase (not decrease) with parameter."
        )

    # Binary search
    lo = param_min
    hi = param_max
    iterations = 0

    for iteration in range(max_iterations):
        iterations = iteration + 1
        mid = (lo + hi) / 2
        proc = build(mid)
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

        # Update bounds (assuming metric decreases with param)
        if current > target.value:
            # Too much privacy loss, need more noise (increase param)
            lo = mid
        else:
            # Too little privacy loss, can use less noise (decrease param)
            hi = mid

    # Max iterations reached - return best estimate
    mid = (lo + hi) / 2
    proc = build(mid)
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
    "epsilon",
    "delta",
    "advantage",
    "beta",
    "risk",
    # Calibration
    "calibrate",
    "CalibrateResult",
]
