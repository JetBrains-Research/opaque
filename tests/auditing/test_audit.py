"""Tests for audit convenience function and AuditResult."""

import numpy as np
import pytest

from opaque.auditing import AuditResult, audit


class TestAudit:
    """Tests for audit convenience function."""

    def test_returns_audit_result(self):
        """Test that audit returns an AuditResult dataclass."""
        in_scores = np.arange(50, 100)
        out_scores = np.arange(0, 50)

        result = audit(in_scores, out_scores, significance=0.05, delta=0)

        assert isinstance(result, AuditResult)
        assert isinstance(result.epsilon, float)
        assert isinstance(result.auroc, float)
        assert isinstance(result.tpr_at_low_fpr, float)
        assert isinstance(result.max_accuracy, float)

    def test_clopper_pearson_method(self):
        """Test with Clopper-Pearson method."""
        in_scores = np.arange(50, 100)
        out_scores = np.arange(0, 50)

        result = audit(in_scores, out_scores, method="clopper_pearson")

        assert result.epsilon > 0
        assert result.auroc > 0.99

    def test_raw_counts_method(self):
        """Test with raw_counts method."""
        in_scores = np.arange(100, 200)
        out_scores = np.arange(0, 100)

        result = audit(in_scores, out_scores, method="raw_counts")

        assert result.epsilon > 0
        assert result.auroc > 0.99

    def test_one_run_method(self):
        """Test with one_run method."""
        in_scores = np.arange(50, 100)
        out_scores = np.arange(0, 50)

        result = audit(in_scores, out_scores, method="one_run")

        assert result.epsilon > 0
        assert result.auroc > 0.99

    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            audit([1, 2], [3, 4], method="invalid")

    def test_frozen(self):
        """Test that AuditResult is immutable."""
        result = audit(np.arange(50, 100), np.arange(0, 50))
        with pytest.raises(AttributeError):
            result.epsilon = 0.0
