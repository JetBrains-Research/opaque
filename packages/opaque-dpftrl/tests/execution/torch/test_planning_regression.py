"""Regression gates for NumPy/SciPy MF planning against Torch autograd."""

from __future__ import annotations

import statistics
import time

import numpy as np
import pytest
from _legacy_planning_reference import legacy_band_mf_coefficients, legacy_blt_loss

from opaque.api.dpftrl.noise import _band_mf
from opaque.api.dpftrl.noise._blt_math import LossFn
from opaque.api.dpftrl.noise._blt_math import loss as blt_loss
from opaque.api.dpftrl.noise._blt_math import optimize as optimize_blt
from opaque.api.dpftrl.noise._blt_math import (
    sensitivity_squared as blt_sensitivity_squared,
)
from opaque.api.dpftrl.noise._toeplitz import loss as toeplitz_loss

pytestmark = pytest.mark.slow


@pytest.mark.parametrize(
    ("n", "min_sep", "max_participations", "momentum", "sensitivity_upper_bound"),
    [
        pytest.param(100, 10, None, 0.9, None, id="momentum-min-sep"),
        pytest.param(100, 10, 1, 1.0, None, id="prefix-min-sep"),
        pytest.param(
            500,
            10,
            1,
            0.9,
            1.2023459630813556,
            id="large-horizon-momentum",
        ),
    ],
)
def test_blt_matches_legacy_autograd_quality(
    n: int,
    min_sep: int,
    max_participations: int | None,
    momentum: float,
    sensitivity_upper_bound: float | None,
) -> None:
    """The production BLT planner must retain the legacy utility level."""
    candidate = optimize_blt(
        n=n,
        min_sep=min_sep,
        max_participations=max_participations,
        workload_coef=np.power(momentum, np.arange(n)),
        max_buffers=3,
    )
    candidate_loss = float(
        blt_loss(
            LossFn.build_min_sep(
                n=n,
                min_sep=min_sep,
                max_participations=max_participations,
                workload_coef=np.power(momentum, np.arange(n)),
            ),
            candidate,
            skip_checks=True,
        )
    )
    reference_loss = legacy_blt_loss(
        n=n,
        min_sep=min_sep,
        max_participations=max_participations,
        momentum=momentum,
    )

    assert np.isfinite(reference_loss)
    assert candidate_loss <= reference_loss * 1.01
    if sensitivity_upper_bound is not None:
        candidate_sensitivity = float(np.sqrt(blt_sensitivity_squared(candidate, n=n)))
        assert candidate_sensitivity <= sensitivity_upper_bound


@pytest.mark.parametrize(
    ("n", "bands"),
    [
        pytest.param(1000, 50, id="n1000-b50"),
        pytest.param(5000, 100, id="n5000-b100"),
    ],
)
def test_band_mf_cold_planning_matches_legacy_autograd_latency(
    n: int, bands: int
) -> None:
    """Cache-miss analytic planning must not regress to finite differences."""
    candidate_times: list[float] = []
    reference_times: list[float] = []
    for run in range(3):
        calls = (
            ("candidate", "reference") if run % 2 == 0 else ("reference", "candidate")
        )
        for call in calls:
            start = time.perf_counter()
            if call == "candidate":
                _band_mf._band_mf_coefficients_cached.cache_clear()
                coefficients = _band_mf._band_mf_coefficients_cached(
                    n, bands, 1.0, None
                )
                candidate_times.append(time.perf_counter() - start)
                assert np.isfinite(toeplitz_loss(coefficients, n=n))
            else:
                coefficients = legacy_band_mf_coefficients(n=n, bands=bands)
                reference_times.append(time.perf_counter() - start)
                assert np.isfinite(toeplitz_loss(coefficients, n=n))

    assert (
        statistics.median(candidate_times) <= statistics.median(reference_times) * 1.25
    )
