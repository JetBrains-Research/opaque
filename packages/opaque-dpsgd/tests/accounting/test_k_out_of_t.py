"""Global k-out-of-t horizon accounting contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import opaque.accounting as acc
import opaque.dpsgd.accounting as dpsgd_acc
from opaque.accounting.types import DpHorizonProcess

if TYPE_CHECKING:
    from collections.abc import Callable

_DELTA = 1e-5


def test_factory_returns_horizon_process():
    process = dpsgd_acc.k_out_of_t(
        dpsgd_acc.gaussian(1.0),
        total_participations=3,
        n_steps=10,
    )
    assert isinstance(process, DpHorizonProcess)
    assert process.n_steps == 10


@pytest.mark.slow
def test_prefixes_are_finite_and_full_matches_pld():
    process = dpsgd_acc.k_out_of_t(
        dpsgd_acc.gaussian(1.0),
        total_participations=3,
        n_steps=10,
    )
    step = acc.horizon_run(process)
    values = [(step * k).epsilon_at(_DELTA) for k in range(1, 11)]
    assert all(value > 0 for value in values)
    assert values[-1] == pytest.approx(process.epsilon_at(_DELTA), rel=1e-12, abs=0)


@pytest.mark.slow
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
    # The first, interior, and full prefixes exercise the distinct horizon
    # paths without rebuilding both expensive PLDs at every step.
    for steps in (1, 4, 8):
        assert k_out.pld_at(steps).epsilon_at(_DELTA) == pytest.approx(
            redrawn.pld_at(steps).epsilon_at(_DELTA),
            rel=1e-12,
            abs=0,
        )


class TestKOutOfTRegressionVectors:
    """Committed ε values for the two supported deterministic inner mechanisms."""

    @pytest.mark.parametrize(
        ("name", "factory", "expected"),
        [
            pytest.param(
                "k_out_of_t(gaussian(1.0), k=2, n_steps=16)",
                lambda: dpsgd_acc.k_out_of_t(
                    dpsgd_acc.gaussian(1.0),
                    total_participations=2,
                    n_steps=16,
                ),
                4.687320185091083,
                id="gaussian",
                marks=pytest.mark.slow,
            ),
            pytest.param(
                "k_out_of_t(adaclip(gaussian(1.1)), k=2, n_steps=16)",
                lambda: dpsgd_acc.k_out_of_t(
                    dpsgd_acc.adaclip(
                        dpsgd_acc.gaussian(1.1),
                        expected_batch_size=250,
                        num_groups=3,
                    ),
                    total_participations=2,
                    n_steps=16,
                ),
                3.9650600884447935,
                id="adaclip",
                marks=pytest.mark.slow,
            ),
        ],
    )
    def test_epsilon_matches_committed_vector(
        self,
        name: str,
        factory: Callable[[], DpHorizonProcess],
        expected: float,
    ):
        delta = 1e-8
        actual = factory().epsilon_at(delta)

        assert actual == pytest.approx(expected, rel=1e-9, abs=3e-9), (
            f"{name}, delta={delta}: epsilon drifted; "
            f"committed={expected:.17g}, observed={actual:.17g}"
        )
