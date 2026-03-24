"""Identity mechanism — zero privacy loss, composition identity element."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .. import opaque_accounting as _native

from opaque_accounting.base import CgfPld, DpProcess, PmfPld
from opaque_accounting.discretization import _make_native_config


@dataclass(frozen=True, slots=True)
class Identity(DpProcess):
    """Identity mechanism — zero privacy loss.

    Identity element of composition:
    ``Identity() | a`` → ``a`` and ``a | Identity()`` → ``a``.
    """

    @functools.lru_cache(maxsize=1)
    def cgf(self) -> CgfPld:
        return CgfPld(_native.cgf_identity_pld())

    def pmf(self, **kwargs: object) -> PmfPld:
        return PmfPld(_native.identity_pld(_make_native_config(**kwargs)))


def identity() -> DpProcess:
    """Identity mechanism (zero privacy loss).

    Useful as a placeholder or identity element in composition.

    Returns:
        An :class:`Identity` process (ε=0 for any δ).
    """
    return Identity()
