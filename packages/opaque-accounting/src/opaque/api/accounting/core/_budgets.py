"""Privacy budgets for calibration and accountant budget tracking.

A **Budget** is a privacy metric paired with a threshold — e.g. "ε ≤ 3 at
δ = 10⁻⁵".  Budgets are used in two places:

1. :func:`~opaque.accounting.calibration.calibrate` — binary search for a
   parameter (e.g. noise_multiplier) that achieves the budget.
2. :class:`~opaque.accounting._accountant.Accountant` — optional budget
   tracking (``acct.budget_exceeded``).

Factory functions (``epsilon_budget``, ``delta_budget``, etc.) return
validated budget instances.

Example::

    from opaque.accounting import budgets

    budget = targets.epsilon_budget(3.0, delta=1e-5)
    budget.evaluate(process)  # → achieved epsilon
    budget.value              # → 3.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from opaque.api.accounting.core._base import DpProcess

# =============================================================================
# Budget protocol
# =============================================================================


class Budget(Protocol):
    """Protocol for privacy budgets.

    A budget defines:
    - **evaluate(process)**: Compute metric value for a process
    - **value**: Budget threshold value to achieve
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


# =============================================================================
# Budget implementations
# =============================================================================


@dataclass(frozen=True, slots=True)
class EpsilonBudget:
    """Budget for (ε, δ)-DP: find noise achieving target epsilon at given delta.

    Parameters must satisfy: epsilon > 0 and delta in (0, 1).
    """

    epsilon: float
    delta: float

    def __post_init__(self) -> None:
        """Validate budget parameters."""
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


@dataclass(frozen=True, slots=True)
class DeltaBudget:
    """Budget for (ε, δ)-DP: find noise achieving target delta at given epsilon.

    Parameters must satisfy: delta in (0, 1) and epsilon > 0.
    """

    delta: float
    epsilon: float

    def __post_init__(self) -> None:
        """Validate budget parameters."""
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


@dataclass(frozen=True, slots=True)
class AdvantageBudget:
    """Budget for f-DP: find noise achieving target advantage.

    Advantage must be in (0, 1) — represents total-variation distance between
    neighboring dataset distributions.
    """

    advantage: float

    def __post_init__(self) -> None:
        """Validate budget parameter."""
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


@dataclass(frozen=True, slots=True)
class BetaBudget:
    """Budget for (α, β) error rates: find noise achieving target beta at given alpha.

    Parameters must satisfy: 0 < alpha < 1 and 0 < beta < 1.
    """

    beta: float
    alpha: float

    def __post_init__(self) -> None:
        """Validate budget parameters."""
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


@dataclass(frozen=True, slots=True)
class RiskBudget:
    """Budget for Bayes risk: find noise achieving target risk at given prior.

    Parameters must satisfy: risk in (0, 1) and prior in (0, 1).
    """

    risk: float
    prior: float

    def __post_init__(self) -> None:
        """Validate budget parameters."""
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
# Budget factories
# =============================================================================


def epsilon_budget(epsilon: float, delta: float) -> EpsilonBudget:
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
    return EpsilonBudget(epsilon=epsilon, delta=delta)


def delta_budget(delta: float, epsilon: float) -> DeltaBudget:
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
    return DeltaBudget(delta=delta, epsilon=epsilon)


def advantage_budget(advantage: float) -> AdvantageBudget:
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
    return AdvantageBudget(advantage=advantage)


def beta_budget(beta: float, alpha: float) -> BetaBudget:
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
    return BetaBudget(beta=beta, alpha=alpha)


def risk_budget(risk: float, prior: float) -> RiskBudget:
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
    return RiskBudget(risk=risk, prior=prior)


__all__ = [
    # Protocol
    "Budget",
    # Budget types
    "EpsilonBudget",
    "DeltaBudget",
    "AdvantageBudget",
    "BetaBudget",
    "RiskBudget",
    # Budget factories
    "epsilon_budget",
    "delta_budget",
    "advantage_budget",
    "beta_budget",
    "risk_budget",
]
