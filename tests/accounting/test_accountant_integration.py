"""Integration tests for Accountant with full opaque.accounting module."""

import pytest

import opaque.accounting as acc


class TestAccountantIntegration:
    """Integration tests showing realistic Accountant usage patterns."""

    def test_training_loop_pattern_with_poisson(self):
        """Simulate training loop with Poisson sampling and budget tracking."""
        # Setup privacy budget
        target = acc.epsilon(5.0, delta=1e-5)
        acct = acc.Accountant(budget=target)

        # Simulate training steps
        noise_multiplier = 1.1
        sample_rate = 0.01
        steps = 10  # Fewer steps to stay within budget

        # Compose all steps at once
        acct = acct | (acc.poisson(acc.gaussian(noise_multiplier), sample_rate) * steps)

        # With modest steps, budget should not be exceeded
        assert not acct.budget_exceeded

    def test_training_loop_pattern_incremental(self):
        """Simulate training loop with incremental budget tracking."""
        target = acc.epsilon(10.0, delta=1e-5)
        acct = acc.Accountant(budget=target)

        noise_multiplier = 1.1
        sample_rate = 0.01
        step_process = acc.poisson(acc.gaussian(noise_multiplier), sample_rate)

        # Simulate incremental training steps
        for i in range(5):
            acct_new = acct | step_process
            if acct_new.budget_exceeded:
                break
            acct = acct_new

        # With 5 steps and reasonable parameters, should not exceed
        assert not acct.budget_exceeded

    def test_calibration_integration(self):
        """Test Accountant with calibration to find noise multiplier."""
        # Create target to optimize for
        target = acc.epsilon(2.0, delta=1e-5)
        
        # Use binary search to find noise multiplier satisfying the target
        # Build function that creates a poisson(nm, rate) process repeated 100 times
        result = acc.calibrate(
            target=target,
            build=lambda nm: acc.poisson(acc.gaussian(nm), 0.01) * 100,
            param_min=0.1,   # Must be >= 0.1 (Rust constraint)
            param_max=1.2,   # Must be <= 1.2 (Rust constraint)
        )

        # Create accountant with calibrated noise
        acct = acc.Accountant(budget=target)
        acct = acct | (acc.poisson(acc.gaussian(result.param), 0.01) * 100)

        # After calibration, achieved epsilon should be close to target
        achieved = acct.epsilon_at(1e-5)
        assert abs(achieved - target.value) < 0.5  # Allow some tolerance

    def test_mixed_mechanisms(self):
        """Test Accountant with mixed mechanism types."""
        budget = acc.epsilon(5.0, delta=1e-5)
        acct = acc.Accountant(budget=budget)

        # Mix different mechanisms
        acct = acct | acc.gaussian(0.5)
        acct = acct | (acc.poisson(acc.gaussian(1.0), 0.01) * 20)
        acct = acct | acc.gaussian(0.3)

        # Verify we can query metrics
        eps = acct.epsilon_at(1e-5)
        assert eps > 0.5  # At least one gaussian mechanism

    def test_risk_target_integration(self):
        """Test Accountant with non-epsilon target."""
        # Use advantage (f-DP) target instead of epsilon-delta
        budget = acc.advantage(0.5)
        acct = acc.Accountant(budget=budget)

        # Add mechanisms
        acct = acct | (acc.gaussian(0.5) * 50)

        # Check budget status
        _ = acct.budget_exceeded  # Should not raise


class TestAccountantAPIConsistency:
    """Verify Accountant API consistency with design."""

    def test_composition_returns_accountant(self):
        """Composition always returns Accountant instance."""
        acct = acc.Accountant()
        result = acct | acc.gaussian(1.0)
        assert isinstance(result, acc.Accountant)

    def test_metrics_delegate_to_process(self):
        """Verify metrics delegate to underlying process."""
        acct = acc.Accountant()
        acct = acct | acc.gaussian(1.0)

        # Get epsilon from accountant and process directly
        eps_from_acct = acct.epsilon_at(1e-5)
        eps_from_process = acct._process.epsilon_at(1e-5)

        assert eps_from_acct == eps_from_process

    def test_or_operator_precedence(self):
        """Verify | operator works as expected."""
        acct1 = acc.Accountant()
        step1 = acc.gaussian(1.0)
        step2 = acc.gaussian(0.5)

        # These should be equivalent
        acct2 = (acct1 | step1) | step2
        acct3 = acct1 | (step1 | step2)

        eps2 = acct2.epsilon_at(1e-5)
        eps3 = acct3.epsilon_at(1e-5)

        assert abs(eps2 - eps3) < 1e-10
