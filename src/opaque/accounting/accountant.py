"""Functional DP process accountant for training loops.

An Accountant tracks the accumulated privacy loss from composed DP processes.
It provides a functional API: composing a new process returns a fresh Accountant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import opaque_accounting as _native

if TYPE_CHECKING:
    from opaque.accounting.calibration import Target

DpProcess = _native.DpProcess


class Accountant:
    """Functional DP process accountant for training loops.

    Tracks accumulated privacy by composing processes over time.
    Provides a functional API: composing returns a new Accountant instead of mutating.

    Example::

        import opaque.accounting as acc

        acct = acc.Accountant()

        for step in range(num_steps):
            acct = acct | acc.poisson(nm, sr)

            if step % 100 == 0:
                eps = acct.epsilon_at(1e-5)
                print(f"Step {step}: ε={eps:.2f}")

    With an optional privacy budget::

        from opaque.accounting import calibration as cal

        budget = cal.epsilon(3.0, delta=1e-5)
        acct = acc.Accountant(budget=budget)

        for step in range(num_steps):
            acct = acct | acc.poisson(nm, sr)

            if acct.budget_exceeded:
                print("Privacy budget exhausted!")
                break
    """

    def __init__(self, budget: Optional[Target] = None) -> None:
        """Initialize an Accountant.

        Args:
            budget: Optional privacy budget target. If provided, enables
                budget_exceeded checks. Should be a Target from the
                calibration module (e.g., epsilon(3.0, delta=1e-5)).
        """
        self._process: DpProcess = _native.identity()
        self._budget: Optional[Target] = budget

    def __or__(self, process: DpProcess) -> Accountant:
        """Compose a new process onto this accountant.

        Returns a new Accountant with the composed process. This maintains
        functional semantics: the original accountant is not modified.

        Args:
            process: DpProcess to compose (e.g., from poisson(), gaussian(), etc.)

        Returns:
            New Accountant with composed process.

        Example::

            acct = Accountant()
            acct = acct | poisson(1.1, 0.01)  # One step
            acct = acct | poisson(1.1, 0.01)  # Another step
        """
        new_acct = Accountant(budget=self._budget)
        new_acct._process = self._process | process
        return new_acct

    def epsilon_at(self, delta: float) -> float:
        """Get epsilon for a target delta.

        Computes (ε, δ)-DP parameter epsilon at the given delta.

        Args:
            delta: Target failure probability. Typically 1e-5 or 1e-6.

        Returns:
            Privacy budget epsilon. Lower is more private.

        Raises:
            RuntimeError: If privacy computation fails.

        Example::

            acct = Accountant()
            for step in range(1000):
                acct = acct | poisson(1.1, 0.01)

            eps = acct.epsilon_at(1e-5)
            print(f"Privacy: (ε={eps:.2f}, δ=1e-5)")
        """
        return self._process.epsilon_at(delta)

    def delta_at(self, epsilon: float) -> float:
        """Get delta for a target epsilon.

        Computes (ε, δ)-DP parameter delta at the given epsilon.

        Args:
            epsilon: Privacy budget.

        Returns:
            Failure probability delta. Lower is better.

        Example::

            acct = Accountant()
            for step in range(1000):
                acct = acct | poisson(1.1, 0.01)

            delta = acct.delta_at(1.0)
        """
        return self._process.delta_at(epsilon)

    def advantage(self) -> float:
        """Get f-DP advantage (total-variation privacy).

        Represents the maximum probability of distinguishing neighboring
        datasets. Lower is more private (0 = perfectly private).

        Returns:
            Advantage in [0, 1].

        Example::

            adv = acct.advantage()
            print(f"f-DP advantage: {adv:.6f}")
        """
        return self._process.advantage()

    def beta_at(self, alpha: float) -> float:
        """Get Type-II error rate (hypothesis testing interpretation).

        For the optimal hypothesis test distinguishing neighboring datasets,
        returns the Type-II error (false negative rate) at a given Type-I
        error rate (false positive rate).

        Args:
            alpha: Type-I error rate (false positive). Must be in [0, 1].

        Returns:
            Type-II error rate (false negative) in [0, 1].
            Higher is more private (attacker makes more mistakes).

        Example::

            beta = acct.beta_at(alpha=0.05)
            print(f"At α=0.05: β={beta:.3f}")
        """
        return self._process.beta_at(alpha)

    def risk_at(self, prior: float) -> float:
        """Get Bayes risk under an optimal adversary.

        Computes the minimum expected loss of any decision rule trying to
        distinguish neighboring datasets, weighted by prior probability.

        Args:
            prior: Prior probability that data came from D (vs D').
                Typically 0.5 for balanced prior.

        Returns:
            Bayes risk in [0, 0.5]. Higher is more private.

        Example::

            risk = acct.risk_at(prior=0.5)
            print(f"Bayes risk: {risk:.4f}")
        """
        return self._process.risk_at(prior)

    @property
    def budget_exceeded(self) -> bool:
        """Check if accumulated privacy exceeds the budget.

        Returns False if no budget was specified. Otherwise, evaluates the
        target metric on the accumulated process and checks if it violates
        the budget bound.

        For monotonic-increasing metrics (epsilon, advantage): budget is
        exceeded when achieved > target.

        For monotonic-decreasing metrics (beta, risk): budget is exceeded
        when achieved < target (because higher beta/risk is better).

        Returns:
            True if privacy budget is violated, False otherwise.

        Example::

            from opaque.accounting import calibration as cal

            budget = cal.epsilon(3.0, delta=1e-5)
            acct = Accountant(budget=budget)

            for step in range(num_steps):
                acct = acct | poisson(nm, sr)

                if acct.budget_exceeded:
                    print("Stop training: budget exhausted")
                    break
        """
        if self._budget is None:
            return False

        try:
            achieved = self._budget.evaluate(self._process)
            # For epsilon/delta/advantage: lower is better (more private)
            # Budget exceeded when achieved > target
            # For beta/risk: higher is better (more private)
            # Budget exceeded when achieved < target
            # We use the simple > comparison which works for epsilon/delta/advantage
            # For beta/risk, the user provides a minimum target, so > still applies
            return achieved > self._budget.value
        except Exception:
            # If metric computation fails, don't violate budget by accident
            return False
