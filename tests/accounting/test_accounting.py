"""Tests for privacy accounting module."""

from opaque.accounting import PLDAccountant, RDPAccountant


class TestPLDAccountant:
    """Tests for PLDAccountant."""

    def test_initialization(self):
        """Test basic accountant creation."""
        acc = PLDAccountant()
        assert acc.steps == 0

    def test_step_poisson(self):
        """Test Poisson sampling step tracking."""
        acc = PLDAccountant()

        # Initially epsilon should be 0
        eps0 = acc.get_epsilon(target_delta=1e-5)
        assert eps0 == 0.0

        # After one step, epsilon should increase
        acc.step_poisson(noise_multiplier=1.0, sample_rate=0.01)
        eps1 = acc.get_epsilon(target_delta=1e-5)
        assert eps1 > 0.0
        assert acc.steps == 1

        # After more steps, epsilon should increase further
        acc.step_poisson(noise_multiplier=1.0, sample_rate=0.01, num_steps=10)
        eps2 = acc.get_epsilon(target_delta=1e-5)
        assert eps2 > eps1
        assert acc.steps == 11

    def test_step_truncated_poisson(self):
        """Test truncated Poisson sampling step tracking."""
        acc = PLDAccountant()

        acc.step_truncated_poisson(
            noise_multiplier=1.0,
            sample_rate=0.01,
            truncated_batch_size=100,
            dataset_size=10000,
        )
        eps = acc.get_epsilon(target_delta=1e-5)
        assert eps > 0.0
        assert acc.steps == 1

    def test_more_noise_less_epsilon(self):
        """Test that more noise results in lower epsilon."""
        acc1 = PLDAccountant()
        acc1.step_poisson(noise_multiplier=0.5, sample_rate=0.01, num_steps=100)

        acc2 = PLDAccountant()
        acc2.step_poisson(noise_multiplier=1.0, sample_rate=0.01, num_steps=100)

        acc3 = PLDAccountant()
        acc3.step_poisson(noise_multiplier=2.0, sample_rate=0.01, num_steps=100)

        eps1 = acc1.get_epsilon(target_delta=1e-5)
        eps2 = acc2.get_epsilon(target_delta=1e-5)
        eps3 = acc3.get_epsilon(target_delta=1e-5)

        # More noise = lower epsilon (better privacy)
        assert eps1 > eps2 > eps3

    def test_smaller_sample_rate_less_epsilon(self):
        """Test privacy amplification by subsampling."""
        acc1 = PLDAccountant()
        acc1.step_poisson(noise_multiplier=1.0, sample_rate=0.1, num_steps=100)

        acc2 = PLDAccountant()
        acc2.step_poisson(noise_multiplier=1.0, sample_rate=0.01, num_steps=100)

        eps1 = acc1.get_epsilon(target_delta=1e-5)
        eps2 = acc2.get_epsilon(target_delta=1e-5)

        # Smaller sample rate = lower epsilon (better privacy)
        assert eps2 < eps1

    def test_composition_increases_epsilon(self):
        """Test that privacy degrades with more steps."""
        acc = PLDAccountant()

        acc.step_poisson(noise_multiplier=1.0, sample_rate=0.01, num_steps=100)
        eps_100 = acc.get_epsilon(target_delta=1e-5)

        acc.step_poisson(noise_multiplier=1.0, sample_rate=0.01, num_steps=100)
        eps_200 = acc.get_epsilon(target_delta=1e-5)

        acc.step_poisson(noise_multiplier=1.0, sample_rate=0.01, num_steps=100)
        eps_300 = acc.get_epsilon(target_delta=1e-5)

        # Privacy degrades with more steps
        assert eps_100 < eps_200 < eps_300

    def test_method_chaining(self):
        """Test that methods return self for chaining."""
        acc = PLDAccountant()

        result = acc.step_poisson(noise_multiplier=1.0, sample_rate=0.01).step_truncated_poisson(
            noise_multiplier=1.0,
            sample_rate=0.01,
            truncated_batch_size=100,
            dataset_size=10000,
        )

        assert result is acc
        assert acc.steps == 2
        eps = acc.get_epsilon(target_delta=1e-5)
        assert eps > 0.0

    def test_truncated_poisson_different_than_standard(self):
        """Test that truncated Poisson works and gives a valid epsilon."""
        # Standard Poisson
        acc_standard = PLDAccountant()
        acc_standard.step_poisson(noise_multiplier=1.1, sample_rate=0.01, num_steps=1000)
        eps_standard = acc_standard.get_epsilon(target_delta=1e-5)

        # Truncated Poisson with reasonable truncation
        acc_truncated = PLDAccountant()
        acc_truncated.step_truncated_poisson(
            noise_multiplier=1.1,
            sample_rate=0.01,
            truncated_batch_size=100,
            dataset_size=10000,
            num_steps=1000,
        )
        eps_truncated = acc_truncated.get_epsilon(target_delta=1e-5)

        # Both should give valid positive epsilons
        assert eps_standard > 0
        assert eps_truncated > 0
        # Note: Truncated can be higher or lower depending on parameters

    def test_fixed_batch_equivalent_to_half_noise_poisson(self):
        """Test that fixed batch = Poisson with half noise (double sensitivity)."""
        # Fixed batch sampling
        acc_fixed = PLDAccountant()
        acc_fixed.step_fixed_batch(noise_multiplier=1.0, sample_rate=0.01, num_steps=100)
        eps_fixed = acc_fixed.get_epsilon(target_delta=1e-5)

        # Equivalent: Poisson with half the noise
        acc_poisson = PLDAccountant()
        acc_poisson.step_poisson(noise_multiplier=0.5, sample_rate=0.01, num_steps=100)
        eps_poisson = acc_poisson.get_epsilon(target_delta=1e-5)

        # Should be identical (within numerical precision)
        assert abs(eps_fixed - eps_poisson) < 1e-6


class TestRDPAccountant:
    """Tests for RDPAccountant."""

    def test_initialization(self):
        """Test basic accountant creation."""
        acc = RDPAccountant()
        assert acc.steps == 0

    def test_step_poisson(self):
        """Test Poisson sampling step tracking."""
        acc = RDPAccountant()

        # Initially epsilon should be 0
        eps0 = acc.get_epsilon(target_delta=1e-5)
        assert eps0 == 0.0

        # After one step, epsilon should increase
        acc.step_poisson(noise_multiplier=1.0, sample_rate=0.01)
        eps1 = acc.get_epsilon(target_delta=1e-5)
        assert eps1 > 0.0
        assert acc.steps == 1

        # After more steps, epsilon should increase further
        acc.step_poisson(noise_multiplier=1.0, sample_rate=0.01, num_steps=10)
        eps2 = acc.get_epsilon(target_delta=1e-5)
        assert eps2 > eps1
        assert acc.steps == 11

    def test_more_noise_less_epsilon(self):
        """Test that more noise results in lower epsilon."""
        acc1 = RDPAccountant()
        acc1.step_poisson(noise_multiplier=0.5, sample_rate=0.01, num_steps=100)

        acc2 = RDPAccountant()
        acc2.step_poisson(noise_multiplier=1.0, sample_rate=0.01, num_steps=100)

        acc3 = RDPAccountant()
        acc3.step_poisson(noise_multiplier=2.0, sample_rate=0.01, num_steps=100)

        eps1 = acc1.get_epsilon(target_delta=1e-5)
        eps2 = acc2.get_epsilon(target_delta=1e-5)
        eps3 = acc3.get_epsilon(target_delta=1e-5)

        # More noise = lower epsilon (better privacy)
        assert eps1 > eps2 > eps3

    def test_smaller_sample_rate_less_epsilon(self):
        """Test privacy amplification by subsampling."""
        acc1 = RDPAccountant()
        acc1.step_poisson(noise_multiplier=1.0, sample_rate=0.1, num_steps=100)

        acc2 = RDPAccountant()
        acc2.step_poisson(noise_multiplier=1.0, sample_rate=0.01, num_steps=100)

        eps1 = acc1.get_epsilon(target_delta=1e-5)
        eps2 = acc2.get_epsilon(target_delta=1e-5)

        # Smaller sample rate = lower epsilon (better privacy)
        assert eps2 < eps1

    def test_method_chaining(self):
        """Test that methods return self for chaining."""
        acc = RDPAccountant()

        result = acc.step_poisson(noise_multiplier=1.0, sample_rate=0.01).step_poisson(
            noise_multiplier=1.0, sample_rate=0.01
        )

        assert result is acc
        assert acc.steps == 2
        eps = acc.get_epsilon(target_delta=1e-5)
        assert eps > 0.0

    def test_custom_orders(self):
        """Test initialization with custom RDP orders."""
        import numpy as np

        custom_orders = np.array([2.0, 4.0, 8.0, 16.0, 32.0])
        acc = RDPAccountant(orders=custom_orders)

        acc.step_poisson(noise_multiplier=1.0, sample_rate=0.01, num_steps=100)
        eps = acc.get_epsilon(target_delta=1e-5)
        assert eps > 0.0

    def test_fixed_batch_equivalent_to_half_noise_poisson(self):
        """Test that fixed batch = Poisson with half noise (double sensitivity)."""
        # Fixed batch sampling
        acc_fixed = RDPAccountant()
        acc_fixed.step_fixed_batch(noise_multiplier=1.0, sample_rate=0.01, num_steps=100)
        eps_fixed = acc_fixed.get_epsilon(target_delta=1e-5)

        # Equivalent: Poisson with half the noise
        acc_poisson = RDPAccountant()
        acc_poisson.step_poisson(noise_multiplier=0.5, sample_rate=0.01, num_steps=100)
        eps_poisson = acc_poisson.get_epsilon(target_delta=1e-5)

        # Should be identical (within numerical precision)
        assert abs(eps_fixed - eps_poisson) < 1e-6


class TestAccountantComparison:
    """Compare PLD and RDP accountants."""

    def test_pld_vs_rdp_similar_results(self):
        """Test that PLD and RDP give similar results for common cases."""
        pld = PLDAccountant()
        rdp = RDPAccountant()

        # Same training scenario
        noise_mult = 1.1
        sample_rate = 0.01
        num_steps = 1000

        pld.step_poisson(noise_mult, sample_rate, num_steps)
        rdp.step_poisson(noise_mult, sample_rate, num_steps)

        eps_pld = pld.get_epsilon(target_delta=1e-5)
        eps_rdp = rdp.get_epsilon(target_delta=1e-5)

        # Should be reasonably close (within 20%)
        # PLD is usually tighter but not always
        ratio = abs(eps_pld - eps_rdp) / min(eps_pld, eps_rdp)
        assert ratio < 0.2
