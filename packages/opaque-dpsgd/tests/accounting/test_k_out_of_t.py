"""Global k-out-of-t horizon accounting contracts."""

from __future__ import annotations

import pytest

import opaque.accounting as acc
import opaque.dpsgd.accounting as dpsgd_acc
from opaque.accounting.types import DpHorizonProcess

_DELTA = 1e-5


def test_factory_returns_horizon_process():
    process = dpsgd_acc.k_out_of_t(
        dpsgd_acc.gaussian(1.0),
        total_participations=3,
        n_steps=10,
    )
    assert isinstance(process, DpHorizonProcess)
    assert process.n_steps == 10


def test_prefixes_are_finite_and_full_matches_pld():
    process = dpsgd_acc.k_out_of_t(
        dpsgd_acc.gaussian(1.0),
        total_participations=3,
        n_steps=10,
    )
    step = acc.per_step(process)
    values = [(step * k).epsilon_at(_DELTA) for k in range(1, 11)]
    assert all(value > 0 for value in values)
    assert values[-1] == pytest.approx(process.epsilon_at(_DELTA), rel=0, abs=0)


def test_k_one_prefix_matches_redrawn_single_epoch_process():
    k_out = dpsgd_acc.k_out_of_t(
        dpsgd_acc.gaussian(1.0),
        total_participations=1,
        n_steps=8,
    )
    redrawn = dpsgd_acc.random_allocation(
        dpsgd_acc.gaussian(1.0),
        num_bins=8,
        n_steps=8,
    )
    for steps in range(1, 9):
        assert k_out.pld_at(steps).epsilon_at(_DELTA) == pytest.approx(
            redrawn.pld_at(steps).epsilon_at(_DELTA),
            rel=0,
            abs=0,
        )
