"""Structural composition nodes: Identity, Composed, Repeated, CachedProcess.

These are the concrete :class:`~opaque.accounting.base.DpProcess` subclasses
that implement the composition algebra.  Mechanism-specific subclasses live in
:mod:`opaque.accounting.types`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import opaque_accounting as _native

from opaque.accounting.base import DiscretizationConfig, DpProcess, Pld


@dataclass(frozen=True, slots=True)
class Identity(DpProcess):
    """Identity mechanism — zero privacy loss.

    Identity element of composition:
    ``Identity() | a`` → ``a`` and ``a | Identity()`` → ``a``.
    """

    config: DiscretizationConfig | None = field(default=None, hash=False, repr=False)

    def pld(self) -> Pld:
        return _native.identity_pld(config=self.config)


@dataclass(frozen=True, slots=True)
class Composed(DpProcess):
    """Heterogeneous composition of two processes."""

    left: DpProcess
    right: DpProcess

    def pld(self) -> Pld:
        return self.left.pld().compose(self.right.pld())


@dataclass(frozen=True, slots=True)
class Repeated(DpProcess):
    """Homogeneous k-fold self-composition."""

    inner: DpProcess
    count: int

    def _leaf_and_count(self) -> tuple[DpProcess, int]:
        return (self.inner, self.count)

    def pld(self) -> Pld:
        return self.inner.pld().self_compose(self.count)


class CachedProcess(DpProcess):
    """Mutable caching wrapper around a :class:`DpProcess`.

    Created by :func:`~opaque.accounting.composition.cached`.
    Computes the PLD on the first :meth:`pld` call and caches it.
    Subsequent calls return the cached result.

    Acts as an **opaque barrier** for merge optimization:
    :meth:`_leaf_and_count` returns ``(self, 1)``, preventing the
    optimizer from looking through the cache boundary.

    This is the only mutable :class:`DpProcess` subclass.
    """

    def __init__(self, inner: DpProcess) -> None:
        self.inner = inner
        self._cached_pld: Pld | None = None

    def pld(self) -> Pld:
        if self._cached_pld is None:
            self._cached_pld = self.inner.pld()
        return self._cached_pld

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return f"CachedProcess({self.inner!r})"
