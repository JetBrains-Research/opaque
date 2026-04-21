"""Caching wrapper and ``cached()`` convenience function."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, overload

from opaque.accounting.base import DpProcess, Pld

if TYPE_CHECKING:
    from opaque.accounting.accountant import Accountant


@dataclass(frozen=True, slots=True)
class CachedProcess(DpProcess):
    """Caching wrapper around a :class:`DpProcess`.

    Created by :func:`cached`.
    Computes the PLD on the first :meth:`pld` call and caches it.
    Subsequent calls return the cached result.

    Acts as an **opaque barrier** for merge optimization:
    :meth:`_leaf_and_count` returns ``(self, 1)``, preventing the
    optimizer from looking through the cache boundary. Cached wrappers
    can still merge via structural equality of their inner processes.

    Note: All DpProcess.pld() methods now have automatic caching.
    This wrapper's primary purpose is to serve as a merge barrier,
    not to add caching (though it does increase the cache size to 16).
    """

    inner: DpProcess

    @functools.lru_cache(maxsize=16)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        pessimistic_estimate: bool | None = None,
        max_grid_size: int | None = None,
    ) -> Pld:
        return self.inner.pld(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            pessimistic_estimate=pessimistic_estimate,
            max_grid_size=max_grid_size,
        )


@overload
def cached(process: Accountant) -> Accountant: ...
@overload
def cached(process: DpProcess) -> CachedProcess: ...


def cached(process: DpProcess | Accountant) -> CachedProcess | Accountant:
    """Wrap a process so that its PLD is computed once and cached.

    Returns a :class:`CachedProcess` that computes the PLD lazily on
    the first :meth:`pld` call and caches the result for all subsequent calls.

    ``CachedProcess`` also acts as an **opaque merge barrier**: the
    composition optimizer will not look through a cached node, so
    the cached PLD is reused as-is during further composition. Cached
    wrappers can still merge via structural equality of their inner
    processes.

    When called on an :class:`~opaque.accounting.accountant.Accountant`,
    returns a new Accountant whose inner process is cached.  Call before
    :meth:`epsilon_at` so that the PLD is populated on the first query
    and reused as an opaque boundary for subsequent composition.

    Example::

        training = acc.cached(acc.poisson(acc.gaussian(1.1), 0.01) * 1000)
        eps = training.epsilon_at(1e-5)   # PLD computed here, cached
        adv = training.advantage()         # reuses cached PLD (free)

    Incremental accounting in a training loop::

        acct = Accountant()
        step = acc.poisson(acc.gaussian(1.1), 0.01)

        for i in range(num_steps):
            acct = acct | step
            if i % eval_interval == 0:
                acct = acc.cached(acct)        # mark as opaque boundary
                eps = acct.epsilon_at(1e-5)    # populates cache
                # next eval only composes delta steps on top of cached PLD

    Args:
        process: The process (or Accountant) to cache.

    Returns:
        A :class:`CachedProcess` wrapping *process*, or a new
        :class:`Accountant` with its inner process cached.
    """
    from opaque.accounting.accountant import Accountant

    if isinstance(process, Accountant):
        new_acct = Accountant(budget=process._budget)
        new_acct._process = cached(process._process)
        return new_acct
    if isinstance(process, CachedProcess):
        return process
    return CachedProcess(process)
