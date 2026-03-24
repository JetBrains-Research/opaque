"""Homogeneous k-fold self-composition of a DP process."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from opaque_accounting.base import CgfPld, DpProcess, PmfPld
from opaque_accounting.discretization import DiscretizationConfig


@dataclass(frozen=True, slots=True)
class Repeated(DpProcess):
    """Homogeneous k-fold self-composition."""

    inner: DpProcess
    count: int

    def _leaf_and_count(self) -> tuple[DpProcess, int]:
        return (self.inner, self.count)

    @functools.lru_cache(maxsize=1)
    def cgf(self) -> CgfPld:
        return self.inner.cgf() * self.count

    @functools.lru_cache(maxsize=8)
    def pmf(self, config: DiscretizationConfig) -> PmfPld:
        return self.inner.pmf(config) * self.count
