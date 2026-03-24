"""Homogeneous k-fold self-composition of a DP process."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from opaque_accounting.base import DpProcess, Pld
from opaque_accounting.discretization import DiscretizationConfig


@dataclass(frozen=True, slots=True)
class Repeated(DpProcess):
    """Homogeneous k-fold self-composition."""

    inner: DpProcess
    count: int

    def _leaf_and_count(self) -> tuple[DpProcess, int]:
        return (self.inner, self.count)

    @functools.lru_cache(maxsize=1)
    def cgf(self) -> Pld:
        return self.inner.cgf().self_compose(self.count)

    @functools.lru_cache(maxsize=8)
    def pmf(self, config: DiscretizationConfig) -> Pld:
        return self.inner.pmf(config).self_compose(self.count)
