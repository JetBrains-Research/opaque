"""Caching wrapper around a DP process."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from opaque.accounting.base import DpProcess, Pld


@dataclass(frozen=True, slots=True)
class CachedProcess(DpProcess):
    """Caching wrapper around a :class:`DpProcess`.

    Created by :func:`~opaque.accounting.composition.cached`.
    Computes the PLD on the first :meth:`pld` call and caches it.
    Subsequent calls return the cached result.

    Acts as an **opaque barrier** for merge optimization:
    :meth:`_leaf_and_count` returns ``(self, 1)``, preventing the
    optimizer from looking through the cache boundary. Cached wrappers
    can still merge via structural equality of their inner processes.
    """

    inner: DpProcess

    @functools.lru_cache(maxsize=1)
    def pld(self) -> Pld:
        return self.inner.pld()

    def state_dict(self) -> dict[str, object]:
        return {
            "type": "CachedProcess",
            "inner": self.inner.state_dict(),
        }

