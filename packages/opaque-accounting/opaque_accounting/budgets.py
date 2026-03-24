"""Privacy budgets for calibration and accountant budget tracking.

A **Budget** is a privacy metric paired with a threshold — e.g. "ε ≤ 3 at
δ = 10⁻⁵".  Budgets are used in two places:

1. :func:`~opaque.accounting.calibration.calibrate` — binary search for a
   parameter (e.g. noise_multiplier) that achieves the budget.
2. :class:`~opaque.accounting.accountant.Accountant` — optional budget
   tracking (``acct.budget_exceeded``).

Factory functions (``epsilon_budget``, ``delta_budget``, etc.) return
validated budget instances.

Example::

    from opaque_accounting import budgets

    budget = budgets.epsilon_budget(3.0, delta=1e-5)
    pld = process.cgf()
    budget.evaluate(pld)  # → achieved epsilon
    budget.value           # → 3.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from opaque_accounting.base import CgfPld, PmfPld

# =============================================================================
# Budget protocol
# =============================================================================


class Budget(Protocol):
    """Protocol for privacy budgets.

    A budget defines:
    - **evaluate(pld)**: Compute metric value for a materialized PLD
    - **value**: Budget threshold value to achieve
    - **name**: Human-readable name for debugging
    - **decreasing**: Whether the metric decreases as the calibrated parameter
      increases.
    """

    value: float
    name: str
    decreasing: bool

    def evaluate(self, pld: CgfPld | PmfPld) -> float:
        """Evaluate the metric on a materialized PLD.

        Args:
            pld: The materialized PLD (from ``process.pmf()`` or
                ``process.cgf()``).

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

    def evaluate(self, pld: CgfPld | PmfPld) -> float:
        """Get epsilon at the target delta."""
        return pld.epsilon_at(self.delta)


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

    def evaluate(self, pld: CgfPld | PmfPld) -> float:
        """Get delta at the target epsilon."""
        return pld.delta_at(self.epsilon)


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

    def evaluate(self, pld: CgfPld | PmfPld) -> float:
        """Get f-DP advantage."""
        return pld.advantage()


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

    def evaluate(self, pld: CgfPld | PmfPld) -> float:
        """Get beta at the target alpha."""
        return pld.beta_at(self.alpha)


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

    def evaluate(self, pld: CgfPld | PmfPld) -> float:
        """Get Bayes risk at the target prior."""
        return pld.risk_at(self.prior)


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
            lambda nm: (acc.poisson(acc.gaussian(nm), 0.01) * 1000).cgf(),
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
    """
    return DeltaBudget(delta=delta, epsilon=epsilon)


def advantage_budget(advantage: float) -> AdvantageBudget:
    """Create a target for f-DP: find noise achieving target advantage.

    Args:
        advantage: Target advantage value (TV distance).

    Returns:
        Calibration target.
    """
    return AdvantageBudget(advantage=advantage)


def beta_budget(beta: float, alpha: float) -> BetaBudget:
    """Create a target for (α, β) error rates: find noise achieving target beta.

    Args:
        beta: Target Type-II error rate.
        alpha: Fixed Type-I error rate.

    Returns:
        Calibration target.
    """
    return BetaBudget(beta=beta, alpha=alpha)


def risk_budget(risk: float, prior: float) -> RiskBudget:
    """Create a target for Bayes risk: find noise achieving target risk.

    Args:
        risk: Target Bayes risk.
        prior: Prior probability of hypothesis H1.

    Returns:
        Calibration target.
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
