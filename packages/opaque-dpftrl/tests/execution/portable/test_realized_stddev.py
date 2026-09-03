"""Provider-neutral realized-noise-stddev identities for DP-FTRL noise."""

from __future__ import annotations

import math

import numpy as np
import pytest

from opaque.api.dpftrl.noise._lambda_cgd import _column_norm
from opaque.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    bsr_strategy,
    identity_strategy,
    lambda_cgd_strategy,
    mf_gaussian_noise,
)
from opaque.random import key
from opaque.types import clipped


def _template(backend_case):
    return {
        "w": backend_case.array(
            np.zeros(8, dtype=np.float32), dtype=backend_case.dtype("float32")
        )
    }


def _step(
    backend_case,
    strategy,
    *,
    n_steps,
    min_sep=1,
    max_participations=None,
    max_norm=1.0,
    noise_multiplier=1.0,
    seed=0,
    n_calls=1,
):
    noise_fn, state = mf_gaussian_noise(
        _template(backend_case),
        strategy,
        n_steps=n_steps,
        min_sep=min_sep,
        max_participations=max_participations,
        noise_multiplier=noise_multiplier,
        key=key(seed),
    )
    grads = clipped(_template(backend_case), max_norm=max_norm)
    realized = []
    for _ in range(n_calls):
        output, state = noise_fn(grads, state)
        realized.append(float(output.noise_stddev))
    return realized


def _row_l2_at_zero(strategy, *, n_steps, min_sep=1, max_participations=None) -> float:
    plan = strategy.execution_plan(
        n_steps=n_steps, min_sep=min_sep, max_participations=max_participations
    )
    return plan.row_l2[0]


def test_identity_realized_stddev_equals_base_sigma(backend_case) -> None:
    sigmas = _step(
        backend_case,
        identity_strategy(),
        n_steps=10,
        noise_multiplier=1.5,
        max_norm=0.7,
        n_calls=4,
    )
    for sigma in sigmas:
        assert sigma == pytest.approx(1.5 * 0.7, rel=1e-12)


@pytest.mark.parametrize(
    ("make_strategy", "parts"),
    [
        (
            lambda: band_mf_strategy(bands=4, momentum=0.9),
            {"n_steps": 20, "min_sep": 1, "max_participations": 20},
        ),
        (
            lambda: blt_strategy(momentum=0.9),
            {"n_steps": 20, "min_sep": 4, "max_participations": 5},
        ),
        (
            lambda: bisr_strategy(bandwidth=4, momentum=0.5),
            {"n_steps": 20, "min_sep": 4, "max_participations": 5},
        ),
        (
            lambda: bsr_strategy(bandwidth=4, alpha=1.0, beta=0.5),
            {"n_steps": 20, "min_sep": 4, "max_participations": 5},
        ),
    ],
    ids=["band_mf", "blt", "bisr", "bsr"],
)
def test_streaming_matrix_realized_stddev_matches_row_l2(
    backend_case, make_strategy, parts
) -> None:
    strategy = make_strategy()
    expected_row_l2 = _row_l2_at_zero(strategy, **parts)
    noise_fn, state = mf_gaussian_noise(
        _template(backend_case),
        strategy,
        **parts,
        noise_multiplier=1.0,
        key=key(0),
    )
    output, _ = noise_fn(clipped(_template(backend_case), max_norm=0.5), state)

    assert output.noise_stddev == pytest.approx(0.5 * expected_row_l2, rel=1e-9)


def test_lambda_cgd_normalized_step_zero_realized_stddev(backend_case) -> None:
    lambda_ = 0.7
    n_steps = 30
    strategy = lambda_cgd_strategy(lambda_=lambda_, normalized=True)
    d_0 = _column_norm(lambda_, n_steps, 0)
    sigmas = _step(
        backend_case,
        strategy,
        n_steps=n_steps,
        noise_multiplier=1.5,
        max_norm=0.4,
        n_calls=1,
    )
    assert sigmas[0] == pytest.approx(1.5 * 0.4 * d_0, rel=1e-12)


def test_lambda_cgd_normalized_step_one_realized_stddev(backend_case) -> None:
    lambda_ = 0.7
    n_steps = 30
    strategy = lambda_cgd_strategy(lambda_=lambda_, normalized=True)
    sigmas = _step(
        backend_case,
        strategy,
        n_steps=n_steps,
        noise_multiplier=1.5,
        max_norm=0.4,
        n_calls=2,
    )
    d_1 = _column_norm(lambda_, n_steps, 1)
    expected = 1.5 * 0.4 * math.sqrt(1.0 + lambda_ * lambda_) * d_1
    assert sigmas[1] == pytest.approx(expected, rel=1e-12)


def test_lambda_cgd_unnormalized_step_one_realized_stddev(backend_case) -> None:
    lambda_ = 0.5
    sigmas = _step(
        backend_case,
        lambda_cgd_strategy(lambda_=lambda_, normalized=False),
        n_steps=20,
        noise_multiplier=1.0,
        max_norm=1.0,
        n_calls=2,
    )
    assert sigmas[0] == pytest.approx(1.0, rel=1e-12)
    assert sigmas[1] == pytest.approx(math.sqrt(1.0 + lambda_ * lambda_), rel=1e-12)


def test_lambda_cgd_zero_reduces_to_iid_realized_stddev(backend_case) -> None:
    sigmas = _step(
        backend_case,
        lambda_cgd_strategy(lambda_=0.0, normalized=False),
        n_steps=20,
        noise_multiplier=1.2,
        max_norm=0.8,
        n_calls=3,
    )
    for sigma in sigmas:
        assert sigma == pytest.approx(1.2 * 0.8, rel=1e-12)


def test_zero_noise_with_infinite_clip_has_zero_realized_stddev(backend_case) -> None:
    sigmas = _step(
        backend_case,
        identity_strategy(),
        n_steps=8,
        noise_multiplier=0.0,
        max_norm=math.inf,
        n_calls=3,
    )
    assert all(sigma == 0.0 for sigma in sigmas)
    assert not any(math.isnan(sigma) for sigma in sigmas)


def test_zero_noise_with_infinite_clip_passes_gradients_through(backend_case) -> None:
    noise_fn, state = mf_gaussian_noise(
        _template(backend_case),
        identity_strategy(),
        n_steps=8,
        noise_multiplier=0.0,
        key=key(0),
    )
    grads = clipped(
        {
            "w": backend_case.array(
                np.ones(8, dtype=np.float32), dtype=backend_case.dtype("float32")
            )
        },
        max_norm=math.inf,
    )
    output, _ = noise_fn(grads, state)

    host = backend_case.to_host(output.pytree["w"])
    assert not np.isnan(host).any()
    np.testing.assert_array_equal(host, np.ones(8, dtype=np.float32))
    assert float(output.noise_stddev) == 0.0
