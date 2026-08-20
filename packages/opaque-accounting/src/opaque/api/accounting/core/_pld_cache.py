"""Shared cache decorators for PLD-producing process methods."""

from __future__ import annotations

import functools
import weakref
from collections import OrderedDict
from collections.abc import Callable, Hashable
from threading import RLock
from typing import TYPE_CHECKING

from .discretization import (
    DiscretizationConfig,
    _use_discretization,
    get_discretization,
)

if TYPE_CHECKING:
    from ._base import Pld

_CacheKey = tuple[DiscretizationConfig, Hashable, int | None]
_MISSING = object()


class _ProcessCache:
    """A weak process reference and its bounded PLD entries."""

    def __init__(self, process_ref: weakref.ReferenceType[object]) -> None:
        self.process_ref = process_ref
        self.entries: OrderedDict[_CacheKey, Pld] = OrderedDict()


class _WeakIdentityPldCache:
    """Keep bounded PLD entries without extending a process's lifetime."""

    def __init__(self, maxsize: int | None) -> None:
        self._maxsize = maxsize
        self._entries: dict[int, _ProcessCache] = {}
        self._shared_entries: OrderedDict[_CacheKey, Pld] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = RLock()
        self._weak_self = weakref.ref(self)

    def _entry_for(self, process: object) -> _ProcessCache:
        process_id = id(process)
        entry = self._entries.get(process_id)
        if entry is not None and entry.process_ref() is process:
            return entry

        cache_ref = self._weak_self

        def remove(ref: weakref.ReferenceType[object]) -> None:
            cache = cache_ref()
            if cache is None:
                return
            with cache._lock:
                current = cache._entries.get(process_id)
                if current is not None and current.process_ref is ref:
                    del cache._entries[process_id]

        process_ref = weakref.ref(process, remove)
        entry = _ProcessCache(process_ref)
        self._entries[process_id] = entry
        return entry

    def _get_shared(self, key: _CacheKey) -> Pld | object:
        cached = self._shared_entries.get(key, _MISSING)
        if cached is not _MISSING:
            self._shared_entries.move_to_end(key)
        return cached

    def _store(
        self, entries: OrderedDict[_CacheKey, Pld], key: _CacheKey, value: Pld
    ) -> None:
        if self._maxsize is None:
            entries[key] = value
        elif self._maxsize > 0:
            entries[key] = value
            if len(entries) > self._maxsize:
                entries.popitem(last=False)

    def get_or_compute(
        self, process: object, key: _CacheKey, compute: Callable[[], Pld]
    ) -> Pld:
        with self._lock:
            entry = self._entry_for(process)
            cached = entry.entries.get(key, _MISSING)
            if cached is not _MISSING:
                entry.entries.move_to_end(key)
                self._hits += 1
                return cached
            cached = self._get_shared(key)
            if cached is not _MISSING:
                self._store(entry.entries, key, cached)
                self._hits += 1
                return cached
            self._misses += 1

        result = compute()

        with self._lock:
            entry = self._entry_for(process)
            cached = entry.entries.get(key, _MISSING)
            if cached is not _MISSING:
                entry.entries.move_to_end(key)
                return cached
            cached = self._get_shared(key)
            if cached is _MISSING:
                self._store(self._shared_entries, key, result)
                cached = result
            self._store(entry.entries, key, cached)
            return cached

    def cache_clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._shared_entries.clear()
            self._hits = 0
            self._misses = 0

    def cache_info(self) -> functools._CacheInfo:
        with self._lock:
            stale_entries = [
                process_id
                for process_id, entry in self._entries.items()
                if entry.process_ref() is None
            ]
            for process_id in stale_entries:
                del self._entries[process_id]
            return functools._CacheInfo(
                self._hits,
                self._misses,
                self._maxsize,
                len(self._shared_entries),
            )


def _resolve_config(
    *,
    discretization: float | None,
    log_x_mass_truncation_bound: float | None,
    max_grid_size: int | None,
    max_conv_grid: int | None,
    num_mc_samples: int | None,
    seed: int | None,
) -> DiscretizationConfig:
    return get_discretization(
        discretization=discretization,
        log_x_mass_truncation_bound=log_x_mass_truncation_bound,
        max_grid_size=max_grid_size,
        max_conv_grid=max_conv_grid,
        num_mc_samples=num_mc_samples,
        seed=seed,
    )


def pld_cache(*, maxsize: int | None):
    """Cache a ``DpProcess.pld`` method by resolved configuration and mechanism."""

    def decorator(method):
        cache = _WeakIdentityPldCache(maxsize)

        @functools.wraps(method)
        def wrapper(
            self,
            *,
            discretization: float | None = None,
            log_x_mass_truncation_bound: float | None = None,
            max_grid_size: int | None = None,
            max_conv_grid: int | None = None,
            num_mc_samples: int | None = None,
            seed: int | None = None,
        ) -> Pld:
            config = _resolve_config(
                discretization=discretization,
                log_x_mass_truncation_bound=log_x_mass_truncation_bound,
                max_grid_size=max_grid_size,
                max_conv_grid=max_conv_grid,
                num_mc_samples=num_mc_samples,
                seed=seed,
            )
            return cache.get_or_compute(
                self,
                (config, self._pld_cache_fingerprint(), None),
                lambda: _compute_pld(method, self, config),
            )

        wrapper.cache_clear = cache.cache_clear
        wrapper.cache_info = cache.cache_info
        return wrapper

    return decorator


def horizon_pld_cache(*, maxsize: int | None):
    """Cache a horizon ``pld_at`` method by configuration and prefix mechanism."""

    def decorator(method):
        cache = _WeakIdentityPldCache(maxsize)

        @functools.wraps(method)
        def wrapper(
            self,
            n_steps: int,
            *,
            discretization: float | None = None,
            log_x_mass_truncation_bound: float | None = None,
            max_grid_size: int | None = None,
            max_conv_grid: int | None = None,
            num_mc_samples: int | None = None,
            seed: int | None = None,
        ) -> Pld:
            if n_steps <= 0 or n_steps > self.n_steps:
                return method(
                    self,
                    n_steps,
                    discretization=discretization,
                    log_x_mass_truncation_bound=log_x_mass_truncation_bound,
                    max_grid_size=max_grid_size,
                    max_conv_grid=max_conv_grid,
                    num_mc_samples=num_mc_samples,
                    seed=seed,
                )
            config = _resolve_config(
                discretization=discretization,
                log_x_mass_truncation_bound=log_x_mass_truncation_bound,
                max_grid_size=max_grid_size,
                max_conv_grid=max_conv_grid,
                num_mc_samples=num_mc_samples,
                seed=seed,
            )
            return cache.get_or_compute(
                self,
                (config, self._pld_cache_fingerprint(n_steps=n_steps), n_steps),
                lambda: _compute_pld(method, self, config, n_steps=n_steps),
            )

        wrapper.cache_clear = cache.cache_clear
        wrapper.cache_info = cache.cache_info
        return wrapper

    return decorator


def _compute_pld(
    method: Callable[..., Pld],
    process: object,
    config: DiscretizationConfig,
    *,
    n_steps: int | None = None,
) -> Pld:
    with _use_discretization(config):
        if n_steps is None:
            return method(process)
        return method(process, n_steps)
