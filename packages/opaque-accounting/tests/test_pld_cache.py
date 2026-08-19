"""Regression tests for resolved PLD cache identity."""

from __future__ import annotations

import pytest

import opaque.accounting as acc


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
