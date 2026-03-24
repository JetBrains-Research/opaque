"""Heterogeneous composition of two DP processes."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from opaque_accounting.base import DpProcess, Pld
from opaque_accounting.discretization import DiscretizationConfig


@dataclass(frozen=True, slots=True)
class Composed(DpProcess):
    """Heterogeneous composition of two processes."""

    left: DpProcess
    right: DpProcess

    @functools.lru_cache(maxsize=1)
    def cgf(self) -> Pld:
        return self.left.cgf().compose(self.right.cgf())

    @functools.lru_cache(maxsize=8)
    def pmf(self, config: DiscretizationConfig) -> Pld:
        left_pld = self.left.pmf(config)
        right_pld = self.right.pmf(config)
        return left_pld.compose(right_pld)
