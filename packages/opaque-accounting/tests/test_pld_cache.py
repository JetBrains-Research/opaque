"""Regression tests for resolved PLD cache identity."""

from __future__ import annotations

import gc
import weakref
from typing import TYPE_CHECKING

import pytest

import opaque.accounting as acc
from opaque.api.accounting.core._pld_cache import horizon_pld_cache, pld_cache

if TYPE_CHECKING:
    from collections.abc import Callable


class _CachedProcess:
    def __init__(self) -> None:
        self.calls = 0

    def _pld_cache_fingerprint(self) -> str:
        return "cached-process"

    @pld_cache(maxsize=2)
    def pld(self) -> object:
        self.calls += 1
        return object()


class _CachedHorizonProcess:
    n_steps = 3

    def __init__(self) -> None:
        self.calls = 0

    def _pld_cache_fingerprint(self, *, n_steps: int) -> tuple[str, int]:
        return ("cached-horizon-process", n_steps)

    @horizon_pld_cache(maxsize=2)
    def pld_at(self, n_steps: int) -> object:
        self.calls += 1
        return object()


class _UnboundedCachedProcess:
    def __init__(self) -> None:
        self.calls = 0

    def _pld_cache_fingerprint(self) -> str:
        return "unbounded-cached-process"

    @pld_cache(maxsize=None)
    def pld(self) -> object:
        self.calls += 1
        return object()


@pytest.fixture(autouse=True)
def _restore_discretization() -> None:
    from opaque.accounting import discretization

    original = discretization._default_config
    try:
        discretization._default_config = None
        yield
    finally:
        discretization._default_config = original


@pytest.mark.parametrize(
    "config",
    [
        {"discretization": 0.2},
        {"tail_mass_truncation": 1e-12},
        {"max_conv_grid": 16},
    ],
)
def test_existing_cached_process_tracks_global_discretization(
    config: dict[str, float | int],
) -> None:
    process = acc.cached(acc.eps_delta(0.11))

    default_pld = process.pld()
    acc.set_discretization(**config)
    changed_pld = process.pld()
    acc.set_discretization()

    assert changed_pld is not default_pld
    assert process.pld() is default_pld


def test_query_override_shares_the_matching_resolved_cache_entry() -> None:
    process = acc.eps_delta(0.11)

    override_pld = process.pld(discretization=0.2)
    acc.set_discretization(discretization=0.2)

    assert process.pld() is override_pld


def test_pld_cache_reuses_entries_evicts_lru_entry_and_clears() -> None:
    process = _CachedProcess()

    first = process.pld(discretization=0.1)
    assert process.pld(discretization=0.1) is first
    process.pld(discretization=0.2)
    process.pld(discretization=0.3)
    assert process.pld(discretization=0.1) is not first
    assert process.calls == 4

    process.pld.cache_clear()
    process.pld(discretization=0.1)
    assert process.calls == 5


def test_pld_cache_reuses_entries_for_equal_processes() -> None:
    _CachedProcess.pld.cache_clear()
    first = _CachedProcess()
    second = _CachedProcess()

    first_pld = first.pld(discretization=0.1)

    assert second.pld(discretization=0.1) is first_pld
    assert first.calls == 1
    assert second.calls == 0


def test_horizon_pld_cache_reuses_entries_and_evicts_lru_entry() -> None:
    process = _CachedHorizonProcess()

    first = process.pld_at(1)
    assert process.pld_at(1) is first
    process.pld_at(2)
    process.pld_at(3)
    assert process.pld_at(1) is not first
    assert process.calls == 4


def test_pld_cache_supports_unbounded_entries() -> None:
    process = _UnboundedCachedProcess()

    first = process.pld(discretization=0.1)
    process.pld(discretization=0.2)

    assert process.pld(discretization=0.1) is first
    assert process.calls == 2


@pytest.mark.parametrize(
    ("process_type", "query"),
    [
        (_CachedProcess, lambda process: process.pld()),
        (_CachedHorizonProcess, lambda process: process.pld_at(1)),
    ],
)
def test_pld_caches_do_not_retain_queried_processes(
    process_type: type[_CachedProcess] | type[_CachedHorizonProcess],
    query: Callable[[_CachedProcess | _CachedHorizonProcess], object],
) -> None:
    process = process_type()
    query(process)
    process_ref = weakref.ref(process)

    del process
    gc.collect()

    assert process_ref() is None


def test_pld_cache_does_not_retain_composed_process_trees() -> None:
    left = acc.eps_delta(0.11)
    right = acc.eps_delta(0.12)
    process = left | right
    process.pld(discretization=0.2)
    process_refs = [weakref.ref(process), weakref.ref(left), weakref.ref(right)]

    del process, left, right
    gc.collect()

    assert all(process_ref() is None for process_ref in process_refs)
