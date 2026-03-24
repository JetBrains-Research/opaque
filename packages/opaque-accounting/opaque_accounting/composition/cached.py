"""Caching wrapper around a DP process."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from opaque_accounting.base import CgfPld, DpProcess, PmfPld
from opaque_accounting.discretization import (
    _DEFAULT_DISCRETIZATION,
    _DEFAULT_LOG_MASS_TRUNCATION_BOUND,
    _DEFAULT_MAX_GRID_SIZE,
    _DEFAULT_PESSIMISTIC_ESTIMATE,
)


@dataclass(frozen=True, slots=True)
class CachedProcess(DpProcess):
    """Caching wrapper around a :class:`DpProcess`.

    Created by :func:`~opaque.accounting.composition.cached`.
    Caches ``pmf()`` and ``cgf()`` results on first call.

    Acts as an **opaque barrier** for merge optimization:
    :meth:`_leaf_and_count` returns ``(self, 1)``, preventing the
    optimizer from looking through the cache boundary. Cached wrappers
    can still merge via structural equality of their inner processes.
    """

    inner: DpProcess

    @functools.lru_cache(maxsize=1)
    def cgf(self) -> CgfPld:
        return self.inner.cgf()

    def pmf(self, **kwargs: object) -> PmfPld:
        # Normalize kwargs to a fixed tuple so lru_cache works.
        key = (
            kwargs.get("discretization", _DEFAULT_DISCRETIZATION),
            kwargs.get("log_mass_truncation_bound", _DEFAULT_LOG_MASS_TRUNCATION_BOUND),
            kwargs.get("pessimistic_estimate", _DEFAULT_PESSIMISTIC_ESTIMATE),
            kwargs.get("max_grid_size", _DEFAULT_MAX_GRID_SIZE),
        )
        return self._pmf_cached(key)

    @functools.lru_cache(maxsize=16)
    def _pmf_cached(self, key: tuple) -> PmfPld:
        d, lmt, pe, mg = key
        return self.inner.pmf(
            discretization=d,
            log_mass_truncation_bound=lmt,
            pessimistic_estimate=pe,
            max_grid_size=mg,
        )
