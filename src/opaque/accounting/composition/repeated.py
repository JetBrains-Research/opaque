"""Homogeneous k-fold self-composition of a DP process."""

from __future__ import annotations

from dataclasses import dataclass

from opaque.accounting.base import DpProcess, Pld


@dataclass(frozen=True, slots=True)
class Repeated(DpProcess):
    """Homogeneous k-fold self-composition."""

    inner: DpProcess
    count: int

    def _leaf_and_count(self) -> tuple[DpProcess, int]:
        return (self.inner, self.count)

    def pld(self) -> Pld:
        return self.inner.pld().self_compose(self.count)
