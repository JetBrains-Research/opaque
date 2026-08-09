"""Iterative repr for arbitrarily-nested ``DpProcess`` composition trees.

Mirrors :mod:`._iter_hash` / :mod:`._iter_eq`: the composition wrappers
delegate ``__repr__`` here so deep chains — left- or right-skewed
``Composed`` spines and the ``cached(acct)``-per-round alternation the
:class:`._cached.CachedProcess` docstring recommends — render without
overflowing the interpreter stack.  Output is string-identical to the
dataclass-generated reprs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opaque.api.accounting.core._base import DpProcess


def iter_repr(process: DpProcess) -> str:
    """Iterative post-order repr of a ``DpProcess`` tree."""
    # Lazy imports break the cycle: each wrapper module imports this
    # helper for its own ``__repr__``.
    from ._cached import CachedProcess
    from ._composed import Composed
    from ._repeated import Repeated

    # id-keyed part strings; plain lookups (never popped) so a child
    # object shared by two parents renders once and serves both.
    parts: dict[int, str] = {}
    stack: list[tuple[DpProcess, bool]] = [(process, False)]
    while stack:
        node, expanded = stack.pop()
        if not expanded and id(node) in parts:
            continue  # shared subtree already rendered
        if isinstance(node, Composed):
            if expanded:
                parts[id(node)] = (
                    f"Composed(left={parts[id(node.left)]}, "
                    f"right={parts[id(node.right)]})"
                )
            else:
                stack.append((node, True))
                stack.append((node.right, False))
                stack.append((node.left, False))
        elif isinstance(node, Repeated):
            if expanded:
                parts[id(node)] = (
                    f"Repeated(inner={parts[id(node.inner)]}, count={node.count!r})"
                )
            else:
                stack.append((node, True))
                stack.append((node.inner, False))
        elif isinstance(node, CachedProcess):
            if expanded:
                parts[id(node)] = f"CachedProcess(inner={parts[id(node.inner)]})"
            else:
                stack.append((node, True))
                stack.append((node.inner, False))
        else:
            parts[id(node)] = repr(node)
    return parts[id(process)]
