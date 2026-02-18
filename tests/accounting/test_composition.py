"""Tests for opaque.accounting composition algebra (operators + functions)."""

import math

import pytest

import opaque.accounting as acc
from opaque.accounting.nodes import Composed, Identity, Repeated


class TestRepeatOperator:
    """step * k and k * step produce Repeated."""

    def test_mul_returns_repeated(self):
        step = acc.gaussian(0.8)
        result = step * 100
        assert isinstance(result, Repeated)
        assert result.count == 100

    def test_rmul_returns_repeated(self):
        step = acc.gaussian(0.8)
        result = 100 * step
        assert isinstance(result, Repeated)
        assert result.count == 100

    def test_mul_and_rmul_agree(self):
        step = acc.poisson(acc.gaussian(0.8), 0.01)
        a = step * 100
        b = 100 * step
        assert a.epsilon_at(1e-5) == pytest.approx(b.epsilon_at(1e-5))

    def test_nested_repeat_flattens(self):
        """(step * 3) * 4 → Repeated(step, 12)."""
        step = acc.gaussian(0.8)
        nested = (step * 3) * 4
        assert isinstance(nested, Repeated)
        assert nested.count == 12
        assert nested.inner is step

    def test_repeat_single(self):
        step = acc.gaussian(0.8)
        r = step * 1
        assert isinstance(r, Repeated)
        assert r.count == 1


class TestComposeOperator:
    """a | b produces Composed (with optimizations)."""

    def test_or_returns_composed(self):
        a = acc.gaussian(0.8)
        b = acc.gaussian(0.5)
        result = a | b
        assert isinstance(result, Composed)

    def test_identity_elision_left(self):
        step = acc.gaussian(0.8)
        result = acc.identity() | step
        assert result is step

    def test_identity_elision_right(self):
        step = acc.gaussian(0.8)
        result = step | acc.identity()
        assert result is step

    def test_direct_merge_same(self):
        """a | a → Repeated(a, 2)."""
        step = acc.gaussian(0.8)
        result = step | step
        assert isinstance(result, Repeated)
        assert result.count == 2

    def test_merge_repeated_same(self):
        """a * n | a * m → Repeated(a, n + m)."""
        step = acc.gaussian(0.8)
        result = (step * 3) | (step * 4)
        assert isinstance(result, Repeated)
        assert result.count == 7

    def test_merge_repeated_left_single_right(self):
        """a * n | a → Repeated(a, n + 1)."""
        step = acc.gaussian(0.8)
        result = (step * 5) | step
        assert isinstance(result, Repeated)
        assert result.count == 6

    def test_merge_single_left_repeated_right(self):
        """a | a * n → Repeated(a, n + 1)."""
        step = acc.gaussian(0.8)
        result = step | (step * 5)
        assert isinstance(result, Repeated)
        assert result.count == 6

    def test_right_spine_merge(self):
        """(X | a * n) | a * m → Composed(X, Repeated(a, n + m))."""
        a = acc.gaussian(0.8)
        x = acc.gaussian(0.5)
        result = (x | (a * 3)) | (a * 4)
        assert isinstance(result, Composed)
        assert isinstance(result.right, Repeated)
        assert result.right.count == 7

    def test_no_merge_different_leaves(self):
        """Different leaves → plain Composed, no merge."""
        a = acc.gaussian(0.8)
        b = acc.gaussian(0.5)
        result = a | b
        assert isinstance(result, Composed)
        assert result.left is a
        assert result.right is b


class TestComposeFunctions:
    """acc.repeat() and acc.compose() match operators."""

    def test_repeat_matches_mul(self):
        step = acc.poisson(acc.gaussian(0.8), 0.01)
        via_op = (step * 100).epsilon_at(1e-5)
        via_fn = acc.repeat(step, 100).epsilon_at(1e-5)
        assert via_op == pytest.approx(via_fn)

    def test_compose_matches_or(self):
        a = acc.gaussian(0.8)
        b = acc.gaussian(0.5)
        via_op = (a | b).epsilon_at(1e-5)
        via_fn = acc.compose(a, b).epsilon_at(1e-5)
        assert via_op == pytest.approx(via_fn)


class TestCompositionProperties:
    """Privacy properties of composition."""

    def test_epsilon_increases_with_composition(self):
        """More steps → higher epsilon."""
        step = acc.poisson(acc.gaussian(0.8), 0.01)
        deltas = [1e-5]
        step_counts = [1, 10, 100, 500]
        for d in deltas:
            epsilons = [(step * k).epsilon_at(d) for k in step_counts]
            for i in range(1, len(epsilons)):
                assert epsilons[i] > epsilons[i - 1]

    def test_epsilon_decreases_with_noise(self):
        """More noise → lower epsilon."""
        sigmas = [0.3, 0.5, 0.8, 1.2]
        epsilons = [
            (acc.poisson(acc.gaussian(s), 0.01) * 100).epsilon_at(1e-5)
            for s in sigmas
        ]
        for i in range(1, len(epsilons)):
            assert epsilons[i] < epsilons[i - 1]

    def test_lower_sample_rate_better_privacy(self):
        """Lower q → lower epsilon (privacy amplification)."""
        rates = [0.01, 0.001, 0.0001]
        epsilons = [
            (acc.poisson(acc.gaussian(0.8), q) * 100).epsilon_at(1e-5)
            for q in rates
        ]
        for i in range(1, len(epsilons)):
            assert epsilons[i] < epsilons[i - 1]

    def test_sublinear_composition_growth(self):
        """10x more steps → < 10x more epsilon (sublinear composition)."""
        step = acc.poisson(acc.gaussian(0.5), 0.01)
        eps_1k = (step * 1000).epsilon_at(1e-5)
        eps_10k = (step * 10000).epsilon_at(1e-5)
        growth = eps_10k / eps_1k
        assert 1.5 < growth < 10.0

    def test_heterogeneous_composition(self):
        """Heterogeneous composition ε > max single phase ε."""
        phase1 = acc.poisson(acc.gaussian(0.5), 0.01) * 100
        phase2 = acc.poisson(acc.gaussian(0.8), 0.01) * 100
        combined = phase1 | phase2
        eps_combined = combined.epsilon_at(1e-5)
        eps1 = phase1.epsilon_at(1e-5)
        eps2 = phase2.epsilon_at(1e-5)
        assert eps_combined > max(eps1, eps2)

    def test_accumulate_in_loop(self):
        """Simulates training loop: repeatedly composing same step merges."""
        step = acc.poisson(acc.gaussian(0.8), 0.01)
        training = acc.identity()
        for _ in range(100):
            training = training | step
        # After identity elision + merge, should be Repeated(step, 100)
        assert isinstance(training, Repeated)
        assert training.count == 100
