"""Functional DP process accountant for training loops.

An Accountant tracks the accumulated privacy loss from composed DP processes.
It provides a functional API: composing a new process returns a fresh Accountant.

Merge optimization is handled entirely by :meth:`DpProcess.__or__`:
identical steps are collapsed using structural equality (``==``), so
``acct | step`` in a loop produces ``Repeated(step, n)`` — one
``self_compose(n)`` (2 FFTs) instead of *n* heterogeneous composes.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from opaque_accounting.base import CgfPld, DpProcess, PmfPld
from opaque_accounting.budgets import (
    AdvantageBudget,
    BetaBudget,
    Budget,
    DeltaBudget,
    EpsilonBudget,
    RiskBudget,
)
from opaque_accounting.discretization import DiscretizationConfig
from opaque_accounting.mechanisms import Identity

__all__ = ["Accountant"]


class Accountant:
    """Functional DP process accountant for training loops.

    Tracks accumulated privacy by composing processes over time.
    Provides a functional API: composing returns a new Accountant instead of mutating.

    Materialize the accumulated process via ``pmf(config)`` or ``cgf()``,
    then query privacy metrics on the returned Pld.

    Example::

        import opaque_accounting as acc

        acct = acc.Accountant()
        step = acc.poisson(acc.gaussian(1.1), 0.01)

        for i in range(num_steps):
            acct = acct | step

            if i % 100 == 0:
                eps = acct.cgf().epsilon_at(1e-5)
                print(f"Step {i}: ε={eps:.2f}")

    With an optional privacy budget::

        from opaque_accounting import calibration as cal

        budget = cal.epsilon_budget(3.0, delta=1e-5)
        acct = acc.Accountant(budget=budget)
        step = acc.poisson(acc.gaussian(1.1), 0.01)

        for i in range(num_steps):
            acct = acct | step
            if acct.budget_exceeded:
                print("Privacy budget exhausted!")
                break
    """

    def __init__(self, budget: Budget | None = None) -> None:
        """Initialize an Accountant.

        Args:
            budget: Optional privacy budget. If provided, enables
                budget_exceeded checks.
        """
        self._process: DpProcess = Identity()
        self._budget: Budget | None = budget

    @property
    def process(self) -> DpProcess:
        """The accumulated DpProcess tree."""
        return self._process

    def __or__(self, process: DpProcess) -> Accountant:
        """Compose a new process onto this accountant.

        Returns a new Accountant with the composed process. The original
        accountant is not modified.

        Args:
            process: DpProcess to compose.

        Returns:
            New Accountant with composed process.
        """
        new_acct = Accountant(budget=self._budget)
        new_acct._process = self._process | process
        return new_acct

    def pmf(self, config: DiscretizationConfig | None = None) -> PmfPld:
        """Materialize the accumulated process as a PMF-backed PLD.

        Args:
            config: Discretization parameters. If None, uses defaults.

        Returns:
            A PmfPld with the full metric suite.
        """
        if config is None:
            config = DiscretizationConfig()
        return self._process.pmf(config)

    def cgf(self) -> CgfPld:
        """Materialize the accumulated process as a CGF-backed PLD.

        Returns:
            A CgfPld with epsilon_at, delta_at, advantage, and pmf().
        """
        return self._process.cgf()

    @property
    def budget_exceeded(self) -> bool:
        """Check if accumulated privacy exceeds the budget.

        Returns False if no budget was specified. Otherwise, materializes
        via CGF (for epsilon/delta/advantage budgets) or PMF (for
        beta/risk budgets) and checks the budget metric.

        Returns:
            True if privacy budget is violated, False otherwise.
        """
        if self._budget is None:
            return False

        needs_pmf = isinstance(self._budget, (BetaBudget, RiskBudget))
        if needs_pmf:
            pld: CgfPld | PmfPld = self._process.pmf(DiscretizationConfig())
        else:
            try:
                pld = self._process.cgf()
            except NotImplementedError:
                pld = self._process.pmf(DiscretizationConfig())
        achieved = self._budget.evaluate(pld)
        return achieved > self._budget.value

    # -- Serialization -------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Serialize accountant state to a plain dict.

        Returns a JSON-compatible dictionary that captures the full process
        tree.  Restore with :meth:`from_state_dict`.
        """
        state: dict[str, Any] = {"process": self._process.state_dict()}
        if self._budget is not None:
            budget_data = dataclasses.asdict(self._budget)
            budget_data["type"] = type(self._budget).__name__
            state["budget"] = budget_data
        return state

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> Accountant:
        """Restore an Accountant from a serialized state dict.

        Args:
            state: Dictionary produced by :meth:`state_dict`.

        Returns:
            Reconstructed Accountant with process and budget restored.
        """
        budget = None
        if "budget" in state:
            budget = _deserialize_budget(state["budget"])
        acct = cls(budget=budget)
        acct._process = DpProcess.from_state_dict(state["process"])
        return acct


_BUDGET_REGISTRY: dict[str, type] = {
    "EpsilonBudget": EpsilonBudget,
    "DeltaBudget": DeltaBudget,
    "AdvantageBudget": AdvantageBudget,
    "BetaBudget": BetaBudget,
    "RiskBudget": RiskBudget,
}


def _deserialize_budget(data: dict[str, Any]) -> Budget:
    data = dict(data)
    type_name = data.pop("type")
    budget_cls = _BUDGET_REGISTRY.get(type_name)
    if budget_cls is None:
        raise ValueError(f"Unknown budget type: {type_name}")
    return budget_cls(**data)
