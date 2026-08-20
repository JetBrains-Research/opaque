"""Regression tests for DP-FTRL PLD cache fingerprints."""

from __future__ import annotations

import opaque.accounting as acc
import opaque.dpftrl.accounting as ftrl_acc
from opaque.dpftrl.noise import band_mf_strategy, blt_strategy, identity_strategy


def _prefix_process(schedule, *, momentum: float = 1.0):
    strategy = band_mf_strategy(bands=4, momentum=momentum, lr_schedule=schedule)
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


def test_distinct_mf_gaussian_parameters_do_not_share_cached_plds() -> None:
    first = ftrl_acc.mf_gaussian(0.0, identity_strategy())
    second = ftrl_acc.mf_gaussian(1.0, identity_strategy())
    first.pld.cache_clear()

    assert first.pld(discretization=0.1) is not second.pld(discretization=0.1)


def test_distinct_strategy_parameters_do_not_share_cached_plds() -> None:
    first = _prefix_process(lambda _step: 1.0, momentum=0.9)
    second = _prefix_process(lambda _step: 1.0, momentum=0.8)
    first.pld.cache_clear()

    assert first.pld(discretization=0.1) is not second.pld(discretization=0.1)


def test_composed_processes_preserve_schedule_cache_identity() -> None:
    constant = acc.eps_delta(0.1) | _prefix_process(lambda _step: 1.0)
    ramp = acc.eps_delta(0.1) | _prefix_process(lambda step: 1.0 + 0.01 * step)

    assert constant == ramp
    assert constant.pld(discretization=0.1) is not ramp.pld(discretization=0.1)


def test_blt_prefix_fingerprint_uses_the_full_schedule() -> None:
    constant = ftrl_acc.balls_in_bins(
        ftrl_acc.mf_gaussian(1.0, blt_strategy(lr_schedule=lambda _step: 1.0)),
        num_bins=4,
        n_steps=16,
    )
    changed_after_prefix = ftrl_acc.balls_in_bins(
        ftrl_acc.mf_gaussian(
            1.0,
            blt_strategy(lr_schedule=lambda step: 1.0 if step < 4 else 0.5),
        ),
        num_bins=4,
        n_steps=16,
    )

    assert constant == changed_after_prefix
    assert constant._pld_cache_key(n_steps=4) != changed_after_prefix._pld_cache_key(
        n_steps=4
    )
