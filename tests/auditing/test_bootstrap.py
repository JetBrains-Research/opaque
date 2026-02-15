"""Tests for bootstrap parameters and resampling."""

import dataclasses

import numpy as np
import pytest

from opaque.auditing import BootstrapParams, attack_auroc, bootstrap


class TestBootstrapParams:
    """Tests for BootstrapParams dataclass."""

    def test_creation_defaults(self):
        """Test creating BootstrapParams with defaults."""
        params = BootstrapParams()
        assert params.num_samples == 1000
        assert params.quantiles == (0.025, 0.975)
        assert params.bias_correction is True
        assert params.acceleration is False
        assert params.seed is None

    def test_creation_custom(self):
        """Test creating BootstrapParams with custom values."""
        params = BootstrapParams(
            num_samples=500,
            quantiles=(0.05, 0.95),
            bias_correction=False,
            seed=42,
        )
        assert params.num_samples == 500
        assert params.quantiles == (0.05, 0.95)
        assert params.bias_correction is False
        assert params.seed == 42

    def test_invalid_num_samples(self):
        """Test that invalid num_samples raises ValueError."""
        with pytest.raises(ValueError, match="num_samples must be positive"):
            BootstrapParams(num_samples=0)
        with pytest.raises(ValueError, match="num_samples must be positive"):
            BootstrapParams(num_samples=-100)

    def test_empty_quantiles(self):
        """Test that empty quantiles raises ValueError."""
        with pytest.raises(ValueError, match="quantiles cannot be empty"):
            BootstrapParams(quantiles=())
        with pytest.raises(ValueError, match="quantiles cannot be empty"):
            BootstrapParams(quantiles=[])

    def test_quantiles_out_of_range(self):
        """Test that quantiles outside (0, 1) raise ValueError."""
        with pytest.raises(ValueError, match="quantiles must be in \\(0, 1\\)"):
            BootstrapParams(quantiles=(0.0, 0.5))
        with pytest.raises(ValueError, match="quantiles must be in \\(0, 1\\)"):
            BootstrapParams(quantiles=(0.5, 1.0))
        with pytest.raises(ValueError, match="quantiles must be in \\(0, 1\\)"):
            BootstrapParams(quantiles=(-0.1, 0.5))
        with pytest.raises(ValueError, match="quantiles must be in \\(0, 1\\)"):
            BootstrapParams(quantiles=(0.5, 1.5))

    def test_acceleration_without_bias_correction(self):
        """Test that acceleration without bias correction raises ValueError."""
        with pytest.raises(
            ValueError, match="Cannot use acceleration without bias correction"
        ):
            BootstrapParams(bias_correction=False, acceleration=True)

    def test_frozen(self):
        """Test that BootstrapParams is frozen (immutable)."""
        params = BootstrapParams()
        with pytest.raises(dataclasses.FrozenInstanceError):
            params.num_samples = 2000

    def test_confidence_interval_factory(self):
        """Test confidence_interval factory method."""
        params = BootstrapParams.confidence_interval(confidence=0.95, seed=42)
        assert params.num_samples == 1000
        assert params.quantiles == pytest.approx((0.025, 0.975))
        assert params.seed == 42

    def test_confidence_interval_90(self):
        """Test 90% confidence interval."""
        params = BootstrapParams.confidence_interval(confidence=0.90)
        expected_quantiles = (0.05, 0.95)
        assert params.quantiles == pytest.approx(expected_quantiles)

    def test_confidence_interval_99(self):
        """Test 99% confidence interval."""
        params = BootstrapParams.confidence_interval(confidence=0.99)
        expected_quantiles = (0.005, 0.995)
        assert params.quantiles == pytest.approx(expected_quantiles)

    def test_confidence_interval_invalid(self):
        """Test that invalid confidence raises ValueError."""
        with pytest.raises(ValueError, match="confidence must be in \\(0, 1\\)"):
            BootstrapParams.confidence_interval(confidence=0.0)
        with pytest.raises(ValueError, match="confidence must be in \\(0, 1\\)"):
            BootstrapParams.confidence_interval(confidence=1.0)
        with pytest.raises(ValueError, match="confidence must be in \\(0, 1\\)"):
            BootstrapParams.confidence_interval(confidence=-0.1)
        with pytest.raises(ValueError, match="confidence must be in \\(0, 1\\)"):
            BootstrapParams.confidence_interval(confidence=1.5)

    def test_quantiles_array_like(self):
        """Test that various array-like quantiles work."""
        # List
        params = BootstrapParams(quantiles=[0.025, 0.975])
        assert len(params.quantiles) == 2

        # Tuple
        params = BootstrapParams(quantiles=(0.1, 0.5, 0.9))
        assert len(params.quantiles) == 3

        # NumPy array
        params = BootstrapParams(quantiles=np.array([0.025, 0.975]))
        assert len(params.quantiles) == 2


class TestBootstrap:
    """Tests for bootstrap function."""

    def test_basic_bootstrap(self):
        """Test basic bootstrap functionality."""
        rng = np.random.default_rng(42)
        in_scores = rng.normal(2.0, 1.0, 100)
        out_scores = rng.normal(0.0, 1.0, 100)

        params = BootstrapParams(num_samples=50, seed=42)
        result = bootstrap(attack_auroc, in_scores, out_scores, params)

        assert isinstance(result, np.ndarray)
        assert len(result) == 2
        assert result[0] < result[1]

    def test_bootstrap_reproducibility(self):
        """Test that bootstrap is reproducible with seed."""
        in_scores = np.arange(50, 100)
        out_scores = np.arange(0, 50)

        params = BootstrapParams(num_samples=20, seed=42)
        result1 = bootstrap(attack_auroc, in_scores, out_scores, params)
        result2 = bootstrap(attack_auroc, in_scores, out_scores, params)

        np.testing.assert_array_equal(result1, result2)

    def test_bootstrap_custom_quantiles(self):
        """Test bootstrap with custom quantiles."""
        rng = np.random.default_rng(42)
        in_scores = rng.normal(2.0, 1.0, 100)
        out_scores = rng.normal(0.0, 1.0, 100)

        params = BootstrapParams(num_samples=50, quantiles=(0.1, 0.5, 0.9), seed=42)
        result = bootstrap(attack_auroc, in_scores, out_scores, params)

        assert len(result) == 3
        assert result[0] <= result[1] <= result[2]
