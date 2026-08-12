"""Heterogeneous composition of two DP processes."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from opaque.api.accounting.core._base import DpProcess, Pld


@dataclass(frozen=True, slots=True, eq=False)
class Composed(DpProcess):
    """Heterogeneous composition of two processes."""

    left: DpProcess
    right: DpProcess

    def __hash__(self) -> int:
        # Iterative tree walk — depth bounded by heap, not stack.
        # See ``_iter_hash``.
        from ._iter_hash import iter_hash

        return iter_hash(self)

    def __eq__(self, other: object) -> bool:
        # Iterative tree walk (dataclass semantics preserved) — depth
        # bounded by heap, not stack.  See ``_iter_eq``.
        if not isinstance(other, DpProcess):
            return NotImplemented
        from ._iter_eq import iter_eq

        return iter_eq(self, other)

    def __repr__(self) -> str:
        # Iterative tree walk, string-identical to the dataclass repr —
        # deep chains (either spine) would otherwise overflow the stack.
        from ._iter_repr import iter_repr

        return iter_repr(self)

    @functools.lru_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        max_conv_grid: int | None = None,
        num_mc_samples: int | None = None,
        seed: int | None = None,
    ) -> Pld:
        kw = {
            "discretization": discretization,
            "log_x_mass_truncation_bound": log_x_mass_truncation_bound,
            "max_grid_size": max_grid_size,
            "max_conv_grid": max_conv_grid,
            "num_mc_samples": num_mc_samples,
            "seed": seed,
        }
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
