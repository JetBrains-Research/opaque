"""Iterative hash for arbitrarily-nested ``DpProcess`` composition trees.

The composition graph grows with every ``acct | step`` accumulation and
``cached(...)`` snapshot. Recursive ``__hash__`` blows the Python call
stack on long training runs because the chain of wrappers each call
``hash(self.inner)`` (or ``hash(node.right)`` / ``hash(node)`` inside
``Composed``), and even though ``Composed.__hash__`` iterates over its
own left spine, the leaves it reaches can be ``Repeated`` /
``CachedProcess`` wrappers whose dataclass-auto-generated ``__hash__``
recursively calls back into ``Composed.__hash__`` on a nested chain —
producing an alternating call-stack cycle that exceeds
``sys.getrecursionlimit()`` after a few thousand DP-SGD steps with
periodic caching.

This module flattens the entire tree into an explicit stack and folds
each node's structural contribution into a running hash without ever
recursing into ``hash(child)`` for wrapper children. Wrapper classes
(``Composed``, ``Repeated``, ``CachedProcess``) delegate ``__hash__``
here; non-wrapper leaves (``GaussianMechanism``, ``Identity``,
``EpsDelta``, ``NonPrivate``, …) are hashed via their own
``__hash__`` (which is safe because they hold no ``DpProcess`` field).

Hashes computed by this helper are consistent with the dataclass
``__eq__`` of each wrapper class: two structurally-equal trees produce
equal hashes, two structurally-distinct trees produce distinct hashes
(modulo standard hash collisions).
"""

from __future__ import annotations

from opaque.api.accounting.core._base import DpProcess

# Sentinel tag strings folded into the hash to distinguish wrapper kinds
# (so e.g. ``Repeated(x, 3)`` doesn't collide with ``Composed(x, x, x)``).
_COMPOSED_TAG = "Composed"
_REPEATED_TAG = "Repeated"
_CACHED_TAG = "Cached"


def iter_hash(process: DpProcess) -> int:
    """Iterative structural hash of a ``DpProcess`` tree.

    Walks the entire tree with an explicit stack — no Python recursion,
    so depth is bounded by available heap rather than
    ``sys.getrecursionlimit()``.
    """
    # Lazy imports break the cycle: each wrapper module imports this
    # helper for its own ``__hash__``.
    from ._cached import CachedProcess
    from ._composed import Composed
    from ._repeated import Repeated

    h = 0
    # Explicit DFS stack. Each entry is a node to visit.
    stack: list[DpProcess] = [process]
    while stack:
        node = stack.pop()
        if isinstance(node, Composed):
            # Fold a structural tag, then schedule the children so the
            # left subtree is processed first (LIFO stack semantics).
            h = hash((h, _COMPOSED_TAG))
            stack.append(node.right)
            stack.append(node.left)
            continue
        if isinstance(node, Repeated):
            h = hash((h, _REPEATED_TAG, node.count))
            stack.append(node.inner)
            continue
        if isinstance(node, CachedProcess):
            h = hash((h, _CACHED_TAG))
            stack.append(node.inner)
            continue
        # Terminal leaf. Its own ``__hash__`` is safe — leaves hold no
        # ``DpProcess`` field, so no further recursion is possible.
        h = hash((h, hash(node)))
    return h
