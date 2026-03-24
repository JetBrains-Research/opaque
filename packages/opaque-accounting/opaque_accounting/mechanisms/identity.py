"""Identity mechanism — zero privacy loss, composition identity element."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .. import opaque_accounting as _native

from opaque_accounting.base import DpProcess, PmfPld
from opaque_accounting.discretization import DiscretizationConfig


@dataclass(frozen=True, slots=True)
class Identity(DpProcess):
    """Identity mechanism — zero privacy loss.

    Identity element of composition:
    ``Identity() | a`` → ``a`` and ``a | Identity()`` → ``a``.
    """

    @functools.lru_cache(maxsize=8)
    def pmf(self, config: DiscretizationConfig) -> PmfPld:
        return PmfPld(_native.identity_pld(config.to_native()))


def identity() -> DpProcess:
    """Identity mechanism (zero privacy loss).

    Useful as a placeholder or identity element in composition.

    Returns:
        An :class:`Identity` process (ε=0 for any δ).
    """
    return Identity()
