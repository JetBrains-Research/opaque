"""Tests for threshold selection strategies."""

import dataclasses

import pytest

from opaque.auditing.threshold import Bonferroni, Explicit, MultiSplit, Split


class TestExplicit:
    """Tests for Explicit threshold strategy."""

    def test_creation(self):
        """Test creating Explicit strategy."""
        strategy = Explicit(threshold=0.5)
        assert strategy.threshold == 0.5

    def test_frozen(self):
        """Test that Explicit is frozen (immutable)."""
        strategy = Explicit(threshold=0.5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            strategy.threshold = 0.7


class TestSplit:
    """Tests for Split threshold strategy."""

    def test_creation_defaults(self):
        """Test creating Split with defaults."""
        strategy = Split()
        assert strategy.threshold_estimation_frac == 0.5
        assert strategy.seed is None

    def test_creation_custom(self):
        """Test creating Split with custom values."""
        strategy = Split(threshold_estimation_frac=0.3, seed=42)
        assert strategy.threshold_estimation_frac == 0.3
        assert strategy.seed == 42

    def test_invalid_fraction(self):
        """Test that invalid fractions raise ValueError."""
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            Split(threshold_estimation_frac=0.0)
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            Split(threshold_estimation_frac=1.0)
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            Split(threshold_estimation_frac=-0.1)
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            Split(threshold_estimation_frac=1.5)

    def test_frozen(self):
        """Test that Split is frozen."""
        strategy = Split(threshold_estimation_frac=0.3)
        with pytest.raises(dataclasses.FrozenInstanceError):
            strategy.threshold_estimation_frac = 0.5


class TestMultiSplit:
    """Tests for MultiSplit threshold strategy."""

    def test_creation_defaults(self):
        """Test creating MultiSplit with defaults."""
        strategy = MultiSplit()
        assert strategy.num_samples == 100
        assert strategy.threshold_estimation_frac == 0.5
        assert strategy.seed is None

    def test_creation_custom(self):
        """Test creating MultiSplit with custom values."""
        strategy = MultiSplit(num_samples=50, threshold_estimation_frac=0.3, seed=42)
        assert strategy.num_samples == 50
        assert strategy.threshold_estimation_frac == 0.3
        assert strategy.seed == 42

    def test_invalid_num_samples(self):
        """Test that invalid num_samples raises ValueError."""
        with pytest.raises(ValueError, match="num_samples must be positive"):
            MultiSplit(num_samples=0)
        with pytest.raises(ValueError, match="num_samples must be positive"):
            MultiSplit(num_samples=-10)

    def test_invalid_fraction(self):
        """Test that invalid fractions raise ValueError."""
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            MultiSplit(threshold_estimation_frac=0.0)
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            MultiSplit(threshold_estimation_frac=1.0)

    def test_frozen(self):
        """Test that MultiSplit is frozen."""
        strategy = MultiSplit(num_samples=50)
        with pytest.raises(dataclasses.FrozenInstanceError):
            strategy.num_samples = 100


class TestBonferroni:
    """Tests for Bonferroni threshold strategy."""

    def test_creation(self):
        """Test creating Bonferroni strategy."""
        strategy = Bonferroni()
        # Should be instantiable with no parameters
        assert isinstance(strategy, Bonferroni)
