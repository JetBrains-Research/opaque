"""Caching wrapper and ``cached()`` convenience function."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, overload

from opaque.api.accounting.core._base import DpProcess, Pld
from opaque.api.accounting.core._pld_cache import pld_cache

from ._per_step import PerStep

if TYPE_CHECKING:
    from opaque.api.accounting.core._accountant import Accountant


@dataclass(frozen=True, slots=True, eq=False)
class CachedProcess(DpProcess):
    """Caching wrapper around a :class:`DpProcess`.

    Created by :func:`cached`.
    Computes the PLD on the first :meth:`pld` call and caches it.
    Subsequent calls return the cached result.

    Acts as an **opaque barrier** for merge optimization, except when
    continuing the same :class:`PerStep` horizon sequence. Cached wrappers
    can still merge via structural equality of their inner processes.

    Since every :meth:`DpProcess.pld` already caches, this wrapper's
    primary purpose is the merge barrier rather than caching (though it
    does raise the cache size to 16).
    """

    inner: DpProcess

    def __hash__(self) -> int:
        # Iterative tree walk — depth bounded by heap, not stack.
        # See ``_iter_hash``.
        from ._iter_hash import iter_hash

        return iter_hash(self)

    def __eq__(self, other: object) -> bool:
        # Iterative tree walk (dataclass semantics preserved) — depth
        # bounded by heap, not stack.  See ``_iter_eq``.
        if not isinstance(other, DpProcess):
            return NotImplemented
        from ._iter_eq import iter_eq

        return iter_eq(self, other)

    def __repr__(self) -> str:
        # Iterative tree walk, string-identical to the dataclass repr.
        from ._iter_repr import iter_repr

        return iter_repr(self)

    def _pld_cache_key(self, *, n_steps: int | None = None) -> tuple[object, ...]:
        from ._iter_cache_key import iter_cache_key

        return iter_cache_key(self, n_steps=n_steps)

    def __or__(self, other: DpProcess) -> DpProcess:
        right_leaf, right_count = other._leaf_and_count()
        if self == right_leaf:
            from ._repeated import Repeated

            return Repeated(self, right_count + 1)

        continued = _continue_horizon(self.inner, other)
        return DpProcess.__or__(self, other) if continued is None else continued

    @pld_cache(maxsize=16)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        max_conv_grid: int | None = None,
        seed: int | None = None,
        mc_resolution: float | None = None,
        mc_failure_probability: float | None = None,
    ) -> Pld:
        return self.inner.pld(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            max_conv_grid=max_conv_grid,
            seed=seed,
            mc_resolution=mc_resolution,
            mc_failure_probability=mc_failure_probability,
        )

    def repeated_pld(
        self,
        count: int,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        max_conv_grid: int | None = None,
        seed: int | None = None,
        mc_resolution: float | None = None,
        mc_failure_probability: float | None = None,
    ) -> Pld:
        return self.inner.repeated_pld(
            count,
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            max_conv_grid=max_conv_grid,
            seed=seed,
            mc_resolution=mc_resolution,
            mc_failure_probability=mc_failure_probability,
        )


def _horizon_group(
    process: DpProcess, *, through_boundary: bool = False
) -> tuple[PerStep, DpProcess, int] | None:
    """Return the horizon step, retained leaf, and repetition count."""
    from ._repeated import Repeated

    node = process
    while through_boundary and isinstance(node, CachedProcess):
        node = node.inner

    if isinstance(node, Repeated):
        leaf, count = node.inner, node.count
    else:
        leaf, count = node, 1

    step = leaf
    while isinstance(step, CachedProcess):
        step = step.inner
    if not isinstance(step, PerStep):
        return None
    return step, leaf, count


def _continue_horizon(prefix: DpProcess, other: DpProcess) -> DpProcess | None:
    """Continue a matching horizon suffix through cached boundaries."""
    from ._composed import Composed
    from ._repeated import Repeated

    next_group = _horizon_group(other)
    if next_group is None:
        return None
    next_step, leaf, next_count = next_group

    node: DpProcess | None = prefix
    count = next_count
    while node is not None:
        cached_node = node if isinstance(node, CachedProcess) else None
        while isinstance(node, CachedProcess):
            node = node.inner

        active_group = _horizon_group(node)
        if active_group is not None:
            active_step, active_leaf, active_count = active_group
            if active_step == next_step:
                leaf = active_leaf
                count += active_count
                node = None
                break
        elif isinstance(node, Composed):
            active_group = _horizon_group(node.right, through_boundary=True)
            if active_group is not None:
                active_step, active_leaf, active_count = active_group
                if active_step == next_step:
                    leaf = active_leaf
                    count += active_count
                    node = node.left
                    continue

        node = cached_node if cached_node is not None else node
        break

    if count == next_count:
        return None

    continued = Repeated(leaf, count)
    if node is None:
        return continued
    cached_prefix = node if isinstance(node, CachedProcess) else CachedProcess(node)
    return Composed(cached_prefix, continued)


@overload
def cached(process: Accountant) -> Accountant: ...
@overload
def cached(process: DpProcess) -> CachedProcess: ...


def cached(process: DpProcess | Accountant) -> CachedProcess | Accountant:
    """Wrap a process so that its PLD is computed once and cached.

    Returns a :class:`CachedProcess` that computes the PLD lazily on
    the first :meth:`pld` call and caches the result for all subsequent calls.

    ``CachedProcess`` also acts as an **opaque merge barrier** for ordinary
    composition. A matching :class:`PerStep` horizon suffix remains
    continuable so its accumulated PLD comes from one ``pld_at(K)`` query.
    Cached wrappers can still merge via structural equality of their inner
    processes.

    When called on an :class:`~opaque.accounting._accountant.Accountant`,
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
    from opaque.api.accounting.core._accountant import Accountant

    match process:
        case Accountant():
            return Accountant(budget=process._budget, prefix=cached(process.process))
        case CachedProcess():
            return process
        case _:
            return CachedProcess(process)
