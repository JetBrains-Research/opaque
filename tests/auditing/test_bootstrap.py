"""Tests for bootstrap parameters."""

import numpy as np
import pytest

from opaque.auditing.bootstrap import BootstrapParams


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
        with pytest.raises(Exception):  # FrozenInstanceError
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
