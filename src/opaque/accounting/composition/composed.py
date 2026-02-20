"""Heterogeneous composition of two DP processes."""

from __future__ import annotations

from dataclasses import dataclass

from opaque.accounting.base import DpProcess, Pld


@dataclass(frozen=True, slots=True)
class Composed(DpProcess):
    """Heterogeneous composition of two processes."""

    left: DpProcess
    right: DpProcess

    def pld(self) -> Pld:
        return self.left.pld().compose(self.right.pld())

    def state_dict(self) -> dict[str, object]:
        return {
            "type": "Composed",
            "left": self.left.state_dict(),
            "right": self.right.state_dict(),
        }
