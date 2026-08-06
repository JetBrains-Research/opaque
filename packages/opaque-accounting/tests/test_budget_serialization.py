"""Budget checkpoint serialization through the public accountant API."""

from __future__ import annotations

import json

import pytest

import opaque.accounting as acc
from opaque.api.accounting.core._accountant import Accountant
from opaque.serialization import from_state_dict, state_dict


class _CustomBudget:
    """Non-dataclass implementation of the public Budget protocol."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.value = threshold
        self.name = "custom"
        self.decreasing = True

    def evaluate(self, process: object) -> float:
        return 0.0


class _UnregisteredBudget(_CustomBudget):
    """Separate protocol implementation that deliberately has no codec."""


def test_registered_non_dataclass_budget_round_trips() -> None:
    acc.register_budget_serializer(
        _CustomBudget,
        lambda budget: {"threshold": budget.threshold},
        lambda state: _CustomBudget(float(state["threshold"])),
    )
    accountant = Accountant(budget=_CustomBudget(2.5))

    checkpoint = json.loads(json.dumps(state_dict(accountant)))
    restored = from_state_dict(Accountant(), checkpoint)

    assert isinstance(restored._budget, _CustomBudget)
    assert restored._budget.threshold == 2.5


@pytest.mark.parametrize(
    "budget",
    [
        acc.epsilon_budget(3.0, delta=1e-5),
        acc.delta_budget(1e-5, epsilon=3.0),
        acc.advantage_budget(0.1),
        acc.beta_budget(0.05, alpha=0.01),
        acc.risk_budget(0.1, prior=0.5),
    ],
)
def test_builtin_budget_round_trips(budget: object) -> None:
    restored = from_state_dict(Accountant(), state_dict(Accountant(budget=budget)))

    assert restored._budget == budget


def test_unregistered_budget_reports_registration_requirement() -> None:
    with pytest.raises(TypeError, match="register_budget_serializer"):
        state_dict(Accountant(budget=_UnregisteredBudget(2.5)))


def test_unknown_budget_checkpoint_type_reports_registration_requirement() -> None:
    serialized = state_dict(Accountant())
    serialized["budget"] = {"type": "example.UnregisteredBudget"}

    with pytest.raises(ValueError, match="no budget serializer is registered"):
        from_state_dict(Accountant(), serialized)
