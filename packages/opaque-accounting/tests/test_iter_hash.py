"""Regression: hashing deeply nested ``DpProcess`` trees doesn't crash.

The previous left-spine-only iteration in ``Composed.__hash__`` worked
for ``acct | step`` accumulation but blew up at scale (4940+ DP-SGD
steps on the V6 SFT long run) because its terminal leaves were
``Repeated`` / ``CachedProcess`` wrappers, whose dataclass-auto
``__hash__`` recursively called back into the next ``Composed`` chain.
The alternating call-stack cycle exceeded ``sys.getrecursionlimit()``.

These tests build the worst-case shapes and assert ``hash(tree)`` is
finite — i.e. no ``RecursionError`` is raised.
"""

from __future__ import annotations

import opaque.accounting as acc
from opaque.api.accounting.core.composition._cached import CachedProcess
from opaque.api.accounting.core.composition._composed import Composed
from opaque.api.accounting.core.composition._repeated import Repeated


def _leaf():
    """A cheap leaf mechanism for tree construction."""
    return acc.identity()


def _build_left_skewed(leaf, depth: int):
    """``Composed(Composed(...Composed(leaf, leaf)..., leaf), leaf)``."""
    chain = leaf
    for _ in range(depth):
        chain = Composed(chain, leaf)
    return chain


# ---------------------------------------------------------------------------
# Recursion-safety regressions
# ---------------------------------------------------------------------------


def test_deeply_nested_composed_hashable():
    """A 10 000-deep left-skewed Composed chain hashes without recursion."""
    chain = _build_left_skewed(_leaf(), 10_000)
    h = hash(chain)
    assert isinstance(h, int)


def test_deeply_nested_cached_inside_composed_hashable():
    """Mirror the V6 SFT pattern: long Composed chain with periodic cached() barriers.

    This is the incremental-accounting pattern recommended by
    ``cached``'s docstring (cache every N steps to compose deltas on
    top of a precomputed PLD). Each ``cached(...)`` wraps the
    accumulated chain into a ``CachedProcess`` whose dataclass-auto
    ``__hash__`` was the culprit in the original bug.
    """
    chain = _leaf()
    for i in range(10_000):
        chain = Composed(chain, _leaf())
        if (i + 1) % 500 == 0:
            chain = acc.cached(chain)
    h = hash(chain)
    assert isinstance(h, int)


def test_deeply_nested_repeated_inside_composed_hashable():
    """A ``Repeated`` whose ``inner`` is itself a deep chain hashes safely."""
    deep_inner = _build_left_skewed(_leaf(), 5_000)
    repeated = Repeated(deep_inner, 100)
    outer_chain = _build_left_skewed(_leaf(), 5_000)
    full = Composed(outer_chain, repeated)
    h = hash(full)
    assert isinstance(h, int)


def test_right_skewed_composed_hashable():
    """Right-skewed chain (less common but possible) also hashes safely.

    The pre-fix ``Composed.__hash__`` only iterated the left spine; a
    pathologically right-skewed tree (``Composed(a, Composed(b,
    Composed(c, ...)))``) would have recursed via ``hash(node.right)``.
    """
    leaf = _leaf()
    chain = leaf
    for _ in range(10_000):
        chain = Composed(leaf, chain)
    h = hash(chain)
    assert isinstance(h, int)


# ---------------------------------------------------------------------------
# Hash / equality contract
# ---------------------------------------------------------------------------


def test_hash_consistent_with_eq_for_composed():
    leaf = _leaf()
    chain1 = _build_left_skewed(leaf, 100)
    chain2 = _build_left_skewed(leaf, 100)
    assert chain1 == chain2
    assert hash(chain1) == hash(chain2)


def test_hash_consistent_with_eq_for_repeated():
    leaf = _leaf()
    r1 = Repeated(leaf, 50)
    r2 = Repeated(leaf, 50)
    assert r1 == r2
    assert hash(r1) == hash(r2)


def test_hash_consistent_with_eq_for_cached():
    leaf = _leaf()
    c1 = CachedProcess(leaf)
    c2 = CachedProcess(leaf)
    assert c1 == c2
    assert hash(c1) == hash(c2)


def test_hash_distinguishes_different_depths():
    """Two chains of different depth should produce different hashes."""
    leaf = _leaf()
    a = _build_left_skewed(leaf, 100)
    b = _build_left_skewed(leaf, 101)
    assert a != b
    assert hash(a) != hash(b)


def test_hash_distinguishes_composed_from_repeated():
    """``Composed(x, x)`` and ``Repeated(x, 2)`` are structurally distinct."""
    leaf = _leaf()
    c = Composed(leaf, leaf)
    r = Repeated(leaf, 2)
    assert c != r
    assert hash(c) != hash(r)
