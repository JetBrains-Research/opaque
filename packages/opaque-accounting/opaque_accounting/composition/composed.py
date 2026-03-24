"""Heterogeneous composition of two DP processes."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from opaque_accounting.base import CgfPld, DpProcess, PmfPld


@dataclass(frozen=True, slots=True)
class Composed(DpProcess):
    """Heterogeneous composition of two processes."""

    left: DpProcess
    right: DpProcess

    @functools.lru_cache(maxsize=1)
    def cgf(self) -> CgfPld:
        return self.left.cgf() | self.right.cgf()

    def pmf(self, **kwargs: object) -> PmfPld:
        return self.left.pmf(**kwargs) | self.right.pmf(**kwargs)
