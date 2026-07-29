"""Heterogeneous composition of two DP processes."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from opaque.api.accounting.core._base import DpProcess, Pld


@dataclass(frozen=True, slots=True)
class Composed(DpProcess):
    """Heterogeneous composition of two processes."""

    left: DpProcess
    right: DpProcess

    def __hash__(self) -> int:
        # Iterative tree walk — depth bounded by heap, not stack.
        # See ``_iter_hash``.
        from ._iter_hash import iter_hash

        return iter_hash(self)

    @functools.lru_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
    ) -> Pld:
        kw = dict(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
        )
        # Walk the left spine iteratively (mirrors __hash__) to avoid
        # O(depth) Python call stack on left-skewed trees.
        rights: list[DpProcess] = []
        node: DpProcess = self
        while isinstance(node, Composed):
            rights.append(node.right)
            node = node.left
        result = node.pld(**kw)
        for right in reversed(rights):
            result = result.compose(right.pld(**kw))
        return result
