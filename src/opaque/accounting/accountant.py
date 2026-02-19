"""Functional DP process accountant for training loops.

An Accountant tracks the accumulated privacy loss from composed DP processes.
It provides a functional API: composing a new process returns a fresh Accountant.

Merge optimization is handled entirely by :meth:`DpProcess.__or__`:
identical steps are collapsed using structural equality (``==``), so
``acct | step`` in a loop produces ``Repeated(step, n)`` — one
``self_compose(n)`` (2 FFTs) instead of *n* heterogeneous composes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opaque.accounting.base import DpProcess
from opaque.accounting.mechanisms import Identity

if TYPE_CHECKING:
    from opaque.accounting.calibration import Budget

__all__ = ["Accountant"]


class Accountant:
    """Functional DP process accountant for training loops.

    Tracks accumulated privacy by composing processes over time.
    Provides a functional API: composing returns a new Accountant instead of mutating.

    Example::

        import opaque.accounting as acc

        acct = acc.Accountant()
        step = acc.poisson(acc.gaussian(1.1), 0.01)

        for i in range(num_steps):
            acct = acct | step

            if i % 100 == 0:
                eps = acct.epsilon_at(1e-5)
                print(f"Step {i}: ε={eps:.2f}")

    With an optional privacy budget::

        from opaque.accounting import calibration as cal

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
                budget_exceeded checks. Should be a Budget from the
                budgets module (e.g., epsilon_budget(3.0, delta=1e-5)).
        """
        self._process: DpProcess = Identity()
        self._budget: Budget | None = budget

    def __or__(self, process: DpProcess) -> Accountant:
        """Compose a new process onto this accountant.

        Returns a new Accountant with the composed process.  The original
        accountant is not modified.

        Merge optimization is automatic: ``DpProcess.__or__`` uses
        structural equality to collapse identical steps into a single
        :class:`~opaque.accounting.composition.Repeated` node.

        Args:
            process: DpProcess to compose (e.g., from poisson(), gaussian(), etc.)

        Returns:
            New Accountant with composed process.

        Example::

            acct = Accountant()
            step = poisson(gaussian(1.1), 0.01)
            acct = acct | step  # One step
            acct = acct | step  # Collapsed into Repeated(step, 2)
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

        Example::

            acct = Accountant()
            step = poisson(gaussian(1.1), 0.01)
            for i in range(1000):
                acct = acct | step

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
        """
        return self._process.delta_at(epsilon)

    def advantage(self) -> float:
        """Get f-DP advantage (total-variation privacy).

        Represents the maximum probability of distinguishing neighboring
        datasets. Lower is more private (0 = perfectly private).

        Returns:
            Advantage in [0, 1].
        """
        return self._process.advantage()

    def beta_at(self, alpha: float) -> float:
        """Get Type-II error rate (hypothesis testing interpretation).

        Args:
            alpha: Type-I error rate (false positive). Must be in [0, 1].

        Returns:
            Type-II error rate (false negative) in [0, 1].
            Higher is more private (attacker makes more mistakes).
        """
        return self._process.beta_at(alpha)

    def risk_at(self, prior: float) -> float:
        """Get Bayes risk under an optimal adversary.

        Args:
            prior: Prior probability that data came from D (vs D').
                Typically 0.5 for balanced prior.

        Returns:
            Bayes risk in [0, 0.5]. Higher is more private.
        """
        return self._process.risk_at(prior)

    @property
    def budget_exceeded(self) -> bool:
        """Check if accumulated privacy exceeds the budget.

        Returns False if no budget was specified. Otherwise, evaluates the
        target metric on the accumulated process and checks if it violates
        the budget bound.

        Returns:
            True if privacy budget is violated, False otherwise.
        """
        if self._budget is None:
            return False

        achieved = self._budget.evaluate(self._process)
        return achieved > self._budget.value

    # -- Serialization -------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Serialize accountant state to a plain dict.

        Returns a JSON-compatible dictionary that captures the full process
        tree.  Restore with :meth:`from_state_dict`.

        Example::

            state = acct.state_dict()
            # ... save to disk, send over network, etc.
            acct2 = Accountant.from_state_dict(state)
            assert acct2.epsilon_at(1e-5) == acct.epsilon_at(1e-5)
        """
        return {"process": _serialize_process(self._process)}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> Accountant:
        """Restore an Accountant from a serialized state dict.

        Args:
            state: Dictionary produced by :meth:`state_dict`.

        Returns:
            Reconstructed Accountant (without budget — attach separately).
        """
        acct = cls()
        acct._process = _deserialize_process(state["process"])
        return acct


# =============================================================================
# Process tree serialization helpers
# =============================================================================


def _serialize_config(
    config: Any,
) -> dict[str, Any] | None:
    """Serialize a DiscretizationConfig to a dict, or None."""
    if config is None:
        return None
    return {
        "discretization": config.discretization,
        "log_mass_truncation_bound": config.log_mass_truncation_bound,
        "pessimistic_estimate": config.pessimistic_estimate,
        "max_grid_size": config.max_grid_size,
    }


def _deserialize_config(data: dict[str, Any] | None) -> Any:
    """Deserialize a DiscretizationConfig from a dict, or return None."""
    if data is None:
        return None
    from opaque.accounting.base import DiscretizationConfig

    return DiscretizationConfig(
        discretization=data["discretization"],
        log_mass_truncation_bound=data["log_mass_truncation_bound"],
        pessimistic_estimate=data["pessimistic_estimate"],
        max_grid_size=data["max_grid_size"],
    )


def _serialize_process(process: DpProcess) -> dict[str, Any]:
    """Recursively serialize a DpProcess tree to a dict."""
    from opaque.accounting.composition import (
        CachedProcess,
        Composed,
        Repeated,
    )
    from opaque.accounting.amplification import (
        Accumulated,
        Poisson,
        TruncatedPoisson,
    )
    from opaque.accounting.mechanisms import (
        EpsDelta,
        Gaussian,
    )

    if isinstance(process, Identity):
        return {"type": "Identity", "config": _serialize_config(process.config)}
    elif isinstance(process, Gaussian):
        return {
            "type": "Gaussian",
            "noise_multiplier": process.noise_multiplier,
            "config": _serialize_config(process.config),
        }
    elif isinstance(process, Poisson):
        return {
            "type": "Poisson",
            "noise_multiplier": process.noise_multiplier,
            "sample_rate": process.sample_rate,
            "config": _serialize_config(process.config),
        }
    elif isinstance(process, TruncatedPoisson):
        return {
            "type": "TruncatedPoisson",
            "noise_multiplier": process.noise_multiplier,
            "sample_rate": process.sample_rate,
            "batch_size_cap": process.batch_size_cap,
            "dataset_size": process.dataset_size,
            "config": _serialize_config(process.config),
        }
    elif isinstance(process, Accumulated):
        return {
            "type": "Accumulated",
            "noise_multiplier": process.noise_multiplier,
            "sample_rate": process.sample_rate,
            "microbatches": process.microbatches,
            "config": _serialize_config(process.config),
        }
    elif isinstance(process, EpsDelta):
        return {
            "type": "EpsDelta",
            "epsilon": process.epsilon,
            "delta": process.delta,
            "config": _serialize_config(process.config),
        }
    elif isinstance(process, Repeated):
        return {
            "type": "Repeated",
            "inner": _serialize_process(process.inner),
            "count": process.count,
        }
    elif isinstance(process, Composed):
        return {
            "type": "Composed",
            "left": _serialize_process(process.left),
            "right": _serialize_process(process.right),
        }
    elif isinstance(process, CachedProcess):
        return {
            "type": "CachedProcess",
            "inner": _serialize_process(process.inner),
        }
    else:
        raise TypeError(f"Cannot serialize {type(process).__name__}")


def _deserialize_process(data: dict[str, Any]) -> DpProcess:
    """Recursively deserialize a DpProcess tree from a dict."""
    from opaque.accounting.composition import (
        CachedProcess,
        Composed,
        Repeated,
    )
    from opaque.accounting.amplification import (
        Accumulated,
        Poisson,
        TruncatedPoisson,
    )
    from opaque.accounting.mechanisms import (
        EpsDelta,
        Gaussian,
    )

    t = data["type"]
    if t == "Identity":
        return Identity(config=_deserialize_config(data.get("config")))
    elif t == "Gaussian":
        return Gaussian(
            noise_multiplier=data["noise_multiplier"],
            config=_deserialize_config(data.get("config")),
        )
    elif t == "Poisson":
        return Poisson(
            noise_multiplier=data["noise_multiplier"],
            sample_rate=data["sample_rate"],
            config=_deserialize_config(data.get("config")),
        )
    elif t == "TruncatedPoisson":
        return TruncatedPoisson(
            noise_multiplier=data["noise_multiplier"],
            sample_rate=data["sample_rate"],
            batch_size_cap=data["batch_size_cap"],
            dataset_size=data["dataset_size"],
            config=_deserialize_config(data.get("config")),
        )
    elif t == "Accumulated":
        return Accumulated(
            noise_multiplier=data["noise_multiplier"],
            sample_rate=data["sample_rate"],
            microbatches=data["microbatches"],
            config=_deserialize_config(data.get("config")),
        )
    elif t == "EpsDelta":
        return EpsDelta(
            epsilon=data["epsilon"],
            delta=data["delta"],
            config=_deserialize_config(data.get("config")),
        )
    elif t == "Repeated":
        return Repeated(
            inner=_deserialize_process(data["inner"]),
            count=data["count"],
        )
    elif t == "Composed":
        return Composed(
            left=_deserialize_process(data["left"]),
            right=_deserialize_process(data["right"]),
        )
    elif t == "CachedProcess":
        return CachedProcess(
            inner=_deserialize_process(data["inner"]),
        )
    else:
        raise ValueError(f"Unknown process type: {t}")
