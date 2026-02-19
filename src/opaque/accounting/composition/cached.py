"""Caching wrapper around a DP process."""

from __future__ import annotations

from opaque.accounting.base import DpProcess, Pld


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
