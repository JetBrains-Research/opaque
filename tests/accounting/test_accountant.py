"""Tests for the Accountant class."""

import pytest

import opaque.accounting as acc
from opaque.accounting.calibration import epsilon


class TestAccountantBasics:
    """Test basic Accountant construction and composition."""

    def test_init_default(self):
        """Accountant initializes with identity process."""
        acct = acc.Accountant()

        # Should have zero privacy cost (identity process)
        eps = acct.epsilon_at(1e-5)
        assert eps < 1e-10

    def test_init_with_budget(self):
        """Accountant accepts optional budget."""
        budget = epsilon(3.0, delta=1e-5)
        acct = acc.Accountant(budget=budget)
        assert acct._budget is not None

    def test_composition_via_or(self):
        """Composing processes via | returns new Accountant."""
        acct1 = acc.Accountant()
        step = acc.poisson(1.1, 0.01)

        acct2 = acct1 | step

        # acct2 is a different object
        assert acct2 is not acct1

        # acct2 has accumulated privacy
        eps1 = acct1.epsilon_at(1e-5)
        eps2 = acct2.epsilon_at(1e-5)
        assert eps2 > eps1


class TestAccountantMetrics:
    """Test privacy metric queries on Accountant."""

    def test_epsilon_at(self):
        """epsilon_at() returns reasonable values."""
        acct = acc.Accountant()
        acct = acct | (acc.poisson(1.1, 0.01) * 100)

        eps = acct.epsilon_at(1e-5)
        assert eps > 0
        assert eps < 100  # Sanity check

    def test_delta_at(self):
        """delta_at() returns reasonable values."""
        acct = acc.Accountant()
        acct = acct | (acc.poisson(1.1, 0.01) * 100)

        delta = acct.delta_at(1.0)
        assert 0 <= delta <= 1

    def test_advantage(self):
        """advantage() returns f-DP advantage."""
        acct = acc.Accountant()
        acct = acct | acc.gaussian(1.0)

        adv = acct.advantage()
        assert 0 <= adv <= 1

    def test_beta_at(self):
        """beta_at() returns Type-II error rate."""
        acct = acc.Accountant()
        acct = acct | (acc.gaussian(1.0) * 10)

        beta = acct.beta_at(0.05)
        assert 0 <= beta <= 1

    def test_risk_at(self):
        """risk_at() returns Bayes risk."""
        acct = acc.Accountant()
        acct = acct | (acc.gaussian(1.0) * 10)

        risk = acct.risk_at(0.5)
        assert 0 <= risk <= 0.5


class TestAccountantBudget:
    """Test budget tracking in Accountant."""

    def test_no_budget(self):
        """Accountant without budget never exceeds."""
        acct = acc.Accountant()
        acct = acct | (acc.gaussian(0.5) * 100)

        assert not acct.budget_exceeded

    def test_budget_not_exceeded(self):
        """budget_exceeded is False when within budget."""
        # Create a very loose budget with high epsilon
        budget = epsilon(100.0, delta=1e-5)
        acct = acc.Accountant(budget=budget)
        # Use gaussian with reasonable noise and few steps
        acct = acct | (acc.gaussian(1.0) * 5)

        assert not acct.budget_exceeded

    def test_budget_exceeded(self):
        """budget_exceeded is True when privacy cost exceeds budget."""
        budget = epsilon(0.1, delta=1e-5)  # Very tight budget
        acct = acc.Accountant(budget=budget)
        acct = acct | (acc.poisson(1.0, 0.01) * 1000)

        # Should exceed the budget
        assert acct.budget_exceeded


class TestAccountantFunctional:
    """Test functional properties of Accountant."""

    def test_composition_immutability(self):
        """Composing doesn't mutate original Accountant."""
        acct1 = acc.Accountant()
        step = acc.poisson(1.1, 0.01)

        eps1_before = acct1.epsilon_at(1e-5)

        acct2 = acct1 | step

        eps1_after = acct1.epsilon_at(1e-5)

        # Original should be unchanged
        assert eps1_before == eps1_after

    def test_chained_composition(self):
        """Can chain multiple compositions."""
        acct = acc.Accountant()
        step = acc.poisson(1.1, 0.01)

        for _ in range(10):
            acct = acct | step

        eps = acct.epsilon_at(1e-5)

        # Should have accumulated privacy loss
        assert eps > 0

    def test_different_mechanisms_compose(self):
        """Can compose different mechanism types."""
        acct = acc.Accountant()
        acct = acct | acc.gaussian(1.0)
        acct = acct | (acc.poisson(1.1, 0.01) * 10)
        acct = acct | acc.gaussian(0.5)

        eps = acct.epsilon_at(1e-5)
        assert eps > 0

    def test_budget_persists_through_composition(self):
        """Budget is preserved when creating new Accountant via composition."""
        budget = epsilon(5.0, delta=1e-5)
        acct1 = acc.Accountant(budget=budget)

        # Compose multiple times
        acct2 = acct1 | acc.gaussian(0.1)
        acct3 = acct2 | acc.gaussian(0.1)

        # All should have the same budget reference
        assert acct1._budget is budget
        assert acct2._budget is budget
        assert acct3._budget is budget
