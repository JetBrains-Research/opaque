"""Identity balls-in-bins epochs reuse the memoised epoch transform (#802).

Horizons snapped to the same ``rounded`` epoch count produce identical
``(sigma_eff, num_bins, 1, config)`` keys; the epoch transform must be
built once across all of them.
"""

from __future__ import annotations

import pytest

import opaque.dpftrl.accounting as ftrl_acc
from opaque.api.accounting.core._random_allocation_cache import (
    clear_random_allocation_caches,
    epoch_pld,
)
from opaque.api.accounting.dpftrl.amplification._balls_in_bins import BallsInBins
from opaque.dpftrl.noise import identity_strategy

_CFG = {"discretization": 0.01, "max_grid_size": 100_000, "max_conv_grid": 128}
_DELTA = 1e-5


@pytest.fixture(autouse=True)
def _reset_caches():
    clear_random_allocation_caches()
    BallsInBins.pld_at.cache_clear()
    yield
    clear_random_allocation_caches()
    BallsInBins.pld_at.cache_clear()


def test_horizons_snapping_to_same_epochs_share_one_epoch_transform():
    process = ftrl_acc.balls_in_bins(
        ftrl_acc.mf_gaussian(1.0, identity_strategy()),
        num_bins=10,
        n_steps=100,
    )

    # n_steps 25 and 30 both snap to rounded=30 → same sigma_eff.
    first = process.pld_at(25, **_CFG).epsilon_at(_DELTA)
    assert epoch_pld.cache_info().misses == 1
    second = process.pld_at(30, **_CFG).epsilon_at(_DELTA)
    assert first == second
    assert epoch_pld.cache_info().misses == 1

    # A distinct snapped epoch count builds exactly one more transform.
    process.pld_at(40, **_CFG)
    assert epoch_pld.cache_info().misses == 2
