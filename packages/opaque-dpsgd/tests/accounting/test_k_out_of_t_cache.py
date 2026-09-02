"""Horizon queries reuse the memoised random-allocation transforms (#802).

Each distinct prefix horizon used to rebuild the O(G²) block epoch transform
from scratch. Across a run's ε probes the block transform must now be
constructed exactly once, and the full-horizon transform once, regardless of
how many distinct horizons are queried.
"""

from __future__ import annotations

import pytest

import opaque.dpsgd.accounting as dpsgd_acc
from opaque.api.accounting.core._random_allocation_cache import (
    clear_random_allocation_caches,
    epoch_pld,
    prefix_pld,
)
from opaque.api.accounting.dpsgd.amplification._k_out_of_t import KOutOfT

_CFG = {"discretization": 0.01, "max_grid_size": 100_000, "max_conv_grid": 256}
_DELTA = 1e-5


@pytest.fixture(autouse=True)
def _reset_caches():
    clear_random_allocation_caches()
    KOutOfT.pld_at.cache_clear()
    yield
    clear_random_allocation_caches()
    KOutOfT.pld_at.cache_clear()


def test_prefix_horizons_do_not_rebuild_block_epoch_transform():
    process = dpsgd_acc.k_out_of_t(
        dpsgd_acc.gaussian(1.0),
        k=4,
        t=100,
        allocation="block",
    )

    first = process.pld_at(30, **_CFG).epsilon_at(_DELTA)
    # One block transform (t=25, k=1) built for the first prefix query.
    assert epoch_pld.cache_info().misses == 1
    assert prefix_pld.cache_info().misses == 1

    for n in (55, 80, 99, 30):
        process.pld_at(n, **_CFG).epsilon_at(_DELTA)

    # Distinct prefix horizons reuse the same block transform. Prefix
    # transforms are distinct per released count (55/80 both leave 5,
    # 99 leaves 24); the repeated n=30 query hits the outer horizon cache.
    assert epoch_pld.cache_info().misses == 1
    assert prefix_pld.cache_info().misses == 2

    # Full horizon is its own transform; repeated queries add nothing.
    eps_full = process.pld_at(100, **_CFG).epsilon_at(_DELTA)
    assert epoch_pld.cache_info().misses == 2
    assert process.pld_at(100, **_CFG).epsilon_at(_DELTA) == eps_full
    assert epoch_pld.cache_info().misses == 2

    # Prefix ε is unchanged by caching (repeat query is bit-identical).
    assert process.pld_at(30, **_CFG).epsilon_at(_DELTA) == first
