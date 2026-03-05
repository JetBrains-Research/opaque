"""Tests for auditing statistical helpers (one_run/stats.py)."""

import numpy as np
import pytest

from opaque.auditing.one_run.stats import (
    log_sub,
    one_run_p_value,
)


class TestLogSub:
    def test_basic_computation(self):
        x = np.log(10)
        y = np.log(3)
        result = log_sub(x, y)
        expected = np.log(10 - 3)
        assert np.isclose(result, expected)

    def test_near_equal_values(self):
        x = np.log(1.0001)
        y = np.log(1.0)
        result = log_sub(x, y)
        expected = np.log(0.0001)
        assert np.isclose(result, expected, rtol=1e-3)

    def test_equal_values(self):
        x = np.log(5)
        y = np.log(5)
        result = log_sub(x, y)
        assert result == -np.inf

    def test_invalid_order(self):
        with pytest.raises(ValueError, match="y must be <= x"):
            log_sub(np.log(3), np.log(10))


class TestOneRunPValue:
    def test_zero_epsilon(self):
        p_value = one_run_p_value(m=100, n_guess=50, n_correct=25, eps=0, delta=0)
        assert 0.4 < p_value < 0.6

    def test_small_epsilon(self):
        p_value = one_run_p_value(m=100, n_guess=50, n_correct=30, eps=0.5, delta=0)
        assert 0.1 < p_value < 1.0

    def test_with_delta(self):
        p_delta0 = one_run_p_value(m=100, n_guess=50, n_correct=30, eps=1.0, delta=0)
        p_delta = one_run_p_value(m=100, n_guess=50, n_correct=30, eps=1.0, delta=0.01)
        assert p_delta >= p_delta0
