"""Contract tests for the random-allocation per-step adapter."""

from __future__ import annotations

import pytest

import opaque.accounting as acc
import opaque.dpsgd.accounting as dpsgd_acc
from opaque.api.accounting.core.composition.types import Repeated
from opaque.dpsgd.accounting.types import PerStepRandomAllocation
from opaque.serialization import from_state_dict, state_dict

_DELTA = 1e-5


def _epoch():
    return dpsgd_acc.random_allocation(dpsgd_acc.gaussian(1.0), num_bins=4)


class TestPerStepRandomAllocation:
    def test_factory_returns_adapter(self):
        step = dpsgd_acc.per_step(_epoch(), n_steps=10)

        assert isinstance(step, PerStepRandomAllocation)
        assert step.n_steps == 10

    @pytest.mark.parametrize(
        ("steps", "charged_epochs"),
        [(1, 1), (4, 1), (5, 2), (8, 2), (9, 3), (10, 3)],
    )
    def test_rounds_partial_epochs_up(self, steps, charged_epochs):
        epoch = _epoch()
        step = dpsgd_acc.per_step(epoch, n_steps=10)

        assert (step * steps).epsilon_at(_DELTA) == pytest.approx(
            (epoch * charged_epochs).epsilon_at(_DELTA),
            rel=0.0,
            abs=0.0,
        )

    def test_accountant_loop_merges_to_repeated(self):
        step = dpsgd_acc.per_step(_epoch(), n_steps=10)
        accountant = acc.Accountant()
        for _ in range(5):
            accountant |= step

        assert isinstance(accountant.process, Repeated)
        assert accountant.epsilon_at(_DELTA) == pytest.approx(
            (step * 5).epsilon_at(_DELTA),
            rel=0.0,
            abs=0.0,
        )

    def test_cached_adapter_relays_repeated_pld(self):
        step = dpsgd_acc.per_step(_epoch(), n_steps=10)

        assert (acc.cached(step) * 5).epsilon_at(_DELTA) == pytest.approx(
            (step * 5).epsilon_at(_DELTA),
            rel=0.0,
            abs=0.0,
        )

    def test_round_trips_through_process_codec(self):
        step = dpsgd_acc.per_step(_epoch(), n_steps=10)

        restored = from_state_dict(step, state_dict(step))

        assert restored == step

    def test_rejects_invalid_process_or_horizon(self):
        with pytest.raises(TypeError, match="RandomAllocation"):
            dpsgd_acc.per_step(dpsgd_acc.gaussian(1.0), n_steps=10)
        with pytest.raises(ValueError, match="n_steps"):
            dpsgd_acc.per_step(_epoch(), n_steps=0)
        with pytest.raises(ValueError, match="exceeds n_steps"):
            (dpsgd_acc.per_step(_epoch(), n_steps=10) * 11).epsilon_at(_DELTA)
