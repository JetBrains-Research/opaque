"""Iterative PLD cache key construction for composition trees."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opaque.api.accounting.core._base import DpProcess


def iter_cache_key(
    process: DpProcess, *, n_steps: int | None = None
) -> tuple[object, ...]:
    """Return a flat, unambiguous cache key without recursive calls."""
    from ._cached import CachedProcess
    from ._composed import Composed
    from ._repeated import Repeated

    parts: list[object] = []
    stack: list[tuple[DpProcess, int | None, int | None]] = [(process, n_steps, None)]
    while stack:
        node, node_steps, repeat_count = stack.pop()
        if isinstance(node, Composed):
            parts.append("Composed")
            # Repeating a composition self-composes its complete PLD; it does
            # not repeat each child independently.
            stack.append((node.right, None, None))
            stack.append((node.left, None, None))
            continue
        if isinstance(node, Repeated):
            parts.extend(("Repeated", node.count))
            stack.append((node.inner, None, node.count))
            continue
        if isinstance(node, CachedProcess):
            parts.append("CachedProcess")
            # CachedProcess.repeated_pld delegates to its inner process, so the
            # repetition identity remains relevant across this wrapper.
            stack.append((node.inner, node_steps, repeat_count))
            continue
        if repeat_count is not None:
            parts.append(node._repeated_pld_cache_key(repeat_count))
        else:
            parts.append(node._pld_cache_key(n_steps=node_steps))
    return tuple(parts)
