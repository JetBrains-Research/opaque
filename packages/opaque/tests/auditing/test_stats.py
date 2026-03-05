"""Tests for auditing statistical helpers (one_run/stats.py)."""


from opaque.auditing.one_run.stats import one_run_p_value


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
