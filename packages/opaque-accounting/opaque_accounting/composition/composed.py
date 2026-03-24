"""Heterogeneous composition of two DP processes."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from opaque_accounting.base import CgfPld, DpProcess, PmfPld
from opaque_accounting.discretization import DiscretizationConfig


@dataclass(frozen=True, slots=True)
class Composed(DpProcess):
    """Heterogeneous composition of two processes."""

    left: DpProcess
    right: DpProcess

    @functools.lru_cache(maxsize=1)
    def cgf(self) -> CgfPld:
        return self.left.cgf() | self.right.cgf()

    @functools.lru_cache(maxsize=8)
    def pmf(self, config: DiscretizationConfig) -> PmfPld:
        return self.left.pmf(config) | self.right.pmf(config)
