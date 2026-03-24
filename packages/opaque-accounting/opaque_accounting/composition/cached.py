"""Caching wrapper around a DP process."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from opaque_accounting.base import DpProcess, Pld
from opaque_accounting.discretization import DiscretizationConfig


@dataclass(frozen=True, slots=True)
class CachedProcess(DpProcess):
    """Caching wrapper around a :class:`DpProcess`.

    Created by :func:`~opaque.accounting.composition.cached`.
    Caches ``pmf(config)`` and ``cgf()`` results on first call.

    Acts as an **opaque barrier** for merge optimization:
    :meth:`_leaf_and_count` returns ``(self, 1)``, preventing the
    optimizer from looking through the cache boundary. Cached wrappers
    can still merge via structural equality of their inner processes.
    """

    inner: DpProcess

    @functools.lru_cache(maxsize=1)
    def cgf(self) -> Pld:
        return self.inner.cgf()

    @functools.lru_cache(maxsize=16)
    def pmf(self, config: DiscretizationConfig) -> Pld:
        return self.inner.pmf(config)
