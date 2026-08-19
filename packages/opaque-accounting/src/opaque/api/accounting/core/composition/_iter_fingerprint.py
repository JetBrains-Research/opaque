"""Iterative PLD cache fingerprinting for composition trees."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opaque.api.accounting.core._base import DpProcess


def iter_fingerprint(
    process: DpProcess, *, n_steps: int | None = None
) -> tuple[object, ...]:
    """Return a flat, unambiguous cache fingerprint without recursive calls."""
    from ._cached import CachedProcess
    from ._composed import Composed
    from ._repeated import Repeated

    parts: list[object] = []
    stack: list[tuple[DpProcess, int | None]] = [(process, n_steps)]
    while stack:
        node, node_steps = stack.pop()
        if isinstance(node, Composed):
            parts.append("Composed")
            stack.append((node.right, None))
            stack.append((node.left, None))
            continue
        if isinstance(node, Repeated):
            parts.extend(("Repeated", node.count))
            stack.append((node.inner, node.count))
            continue
        if isinstance(node, CachedProcess):
            parts.append("CachedProcess")
            stack.append((node.inner, node_steps))
            continue
        parts.append(node._pld_cache_fingerprint(n_steps=node_steps))
    return tuple(parts)
