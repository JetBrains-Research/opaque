"""Tests for opaque.accounting.nodes — structural composition nodes."""

import math
from dataclasses import FrozenInstanceError

import pytest

import opaque.accounting as acc
from opaque.accounting.base import DpProcess
from opaque.accounting.nodes import CachedProcess, Composed, Identity, Repeated


class TestIdentity:
    """Identity node — zero privacy loss, composition identity element."""

    def test_is_dp_process(self):
        assert isinstance(Identity(), DpProcess)

    def test_frozen(self):
        i = Identity()
        with pytest.raises(FrozenInstanceError):
            i.config = None  # type: ignore[misc]

    def test_zero_epsilon(self):
        eps = Identity().epsilon_at(1e-5)
        assert eps == pytest.approx(0.0, abs=1e-10)

    def test_zero_advantage(self):
        adv = Identity().advantage()
        assert adv == pytest.approx(0.0, abs=1e-10)

    def test_compose_identity_left(self):
        """Identity | a → a."""
        step = acc.gaussian(0.8)
        result = Identity() | step
        assert result is step

    def test_compose_identity_right(self):
        """a | Identity → a."""
        step = acc.gaussian(0.8)
        result = step | Identity()
        assert result is step

    def test_equality(self):
        assert Identity() == Identity()


class TestComposed:
    """Composed node — heterogeneous composition."""

    def test_fields(self):
        a = acc.gaussian(0.8)
        b = acc.gaussian(0.5)
        c = Composed(a, b)
        assert c.left is a
        assert c.right is b

    def test_frozen(self):
        c = Composed(acc.gaussian(0.8), acc.gaussian(0.5))
        with pytest.raises(FrozenInstanceError):
            c.left = acc.gaussian(1.0)  # type: ignore[misc]

    def test_is_dp_process(self):
        c = Composed(acc.gaussian(0.8), acc.gaussian(0.5))
        assert isinstance(c, DpProcess)

    def test_pld_composes(self):
        """PLD should compose left and right."""
        a = acc.gaussian(0.8)
        b = acc.gaussian(0.5)
        composed = Composed(a, b)
        eps = composed.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0
        # eps of composition > eps of each part
        assert eps > a.epsilon_at(1e-5)
        assert eps > b.epsilon_at(1e-5)


class TestRepeated:
    """Repeated node — homogeneous k-fold composition."""

    def test_fields(self):
        step = acc.gaussian(0.8)
        r = Repeated(step, 100)
        assert r.inner is step
        assert r.count == 100

    def test_frozen(self):
        r = Repeated(acc.gaussian(0.8), 100)
        with pytest.raises(FrozenInstanceError):
            r.count = 200  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(Repeated(acc.gaussian(0.8), 10), DpProcess)

    def test_leaf_and_count(self):
        step = acc.gaussian(0.8)
        r = Repeated(step, 100)
        leaf, count = r._leaf_and_count()
        assert leaf is step
        assert count == 100

    def test_pld_self_composes(self):
        step = acc.gaussian(0.8)
        r = Repeated(step, 10)
        eps = r.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0
        # 10x composition > single step
        assert eps > step.epsilon_at(1e-5)


class TestCachedProcess:
    """CachedProcess — mutable caching wrapper."""

    def test_identity_equality(self):
        """Two CachedProcesses with same inner are NOT equal."""
        inner = acc.gaussian(0.8)
        a = CachedProcess(inner)
        b = CachedProcess(inner)
        assert a != b
        assert a == a  # identity

    def test_hash_is_id(self):
        cp = CachedProcess(acc.gaussian(0.8))
        assert hash(cp) == id(cp)

    def test_caches_pld(self):
        """Second pld() call returns cached result."""
        cp = CachedProcess(acc.gaussian(0.8))
        pld1 = cp.pld()
        pld2 = cp.pld()
        assert pld1 is pld2

    def test_pld_returns_valid(self):
        cp = CachedProcess(acc.gaussian(0.8))
        eps = cp.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_opaque_merge_barrier(self):
        """CachedProcess prevents merge optimization."""
        step = acc.gaussian(0.8)
        cached_a = CachedProcess(step)
        cached_b = CachedProcess(step)
        result = cached_a | cached_b
        # Should NOT merge into Repeated because CachedProcess uses identity equality
        assert isinstance(result, Composed)

    def test_repr(self):
        cp = CachedProcess(acc.gaussian(0.8))
        assert "CachedProcess" in repr(cp)


class TestCachedFunction:
    """acc.cached() convenience function."""

    def test_wraps_process(self):
        step = acc.poisson(acc.gaussian(0.8), 0.01) * 100
        cached_step = acc.cached(step)
        assert isinstance(cached_step, CachedProcess)

    def test_idempotent(self):
        """cached(cached(x)) returns the same CachedProcess."""
        step = acc.gaussian(0.8)
        c1 = acc.cached(step)
        c2 = acc.cached(c1)
        assert c2 is c1
