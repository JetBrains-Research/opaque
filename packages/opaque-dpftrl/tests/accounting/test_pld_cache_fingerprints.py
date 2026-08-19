"""Regression tests for DP-FTRL PLD cache fingerprints."""

from __future__ import annotations

import opaque.accounting as acc
import opaque.dpftrl.accounting as ftrl_acc
from opaque.dpftrl.noise import band_mf_strategy


def _prefix_process(schedule):
    strategy = band_mf_strategy(bands=4, lr_schedule=schedule)
    process = ftrl_acc.poisson(
        ftrl_acc.mf_gaussian(1.0, strategy),
        sample_rate=0.01,
        n_steps=16,
    )
    return acc.cached(acc.per_step(process)) * 8


def test_distinct_materialized_schedules_do_not_share_cached_plds() -> None:
    constant = _prefix_process(lambda _step: 1.0)
    ramp = _prefix_process(lambda step: 1.0 + 0.01 * step)

    assert constant == ramp
    assert constant.pld(discretization=0.1) is not ramp.pld(discretization=0.1)


def test_equal_materialized_schedules_reuse_cached_plds() -> None:
    first = _prefix_process(lambda step: 1.0 + 0.01 * step)
    second = _prefix_process(lambda step: float(1.0 + 0.01 * step))

    assert first == second
    assert first.pld(discretization=0.1) is second.pld(discretization=0.1)


def test_composed_processes_preserve_schedule_cache_identity() -> None:
    constant = acc.eps_delta(0.1) | _prefix_process(lambda _step: 1.0)
    ramp = acc.eps_delta(0.1) | _prefix_process(lambda step: 1.0 + 0.01 * step)

    assert constant == ramp
    assert constant.pld(discretization=0.1) is not ramp.pld(discretization=0.1)
