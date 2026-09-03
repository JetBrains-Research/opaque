"""Portable coverage for public DP-FTRL mechanisms and samplers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from opaque import ops
from opaque.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    bsr_strategy,
    identity_strategy,
    lambda_cgd_strategy,
    mf_gaussian_noise,
)
from opaque.dpftrl.sampling import (
    BallsInBinsSampler,
    BMinSepSampler,
    CyclicPoissonSampler,
    SequentialBatchSampler,
)
from opaque.random import key
from opaque.serialization import from_state_dict, state_dict
from opaque.types import (
    PerGroup,
    SecondMomentClippingOutput,
    SecondMomentNoiseOutput,
    clipped,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_N_STEPS = 4
_STRATEGIES = [
    ("identity", identity_strategy()),
    ("band_mf", band_mf_strategy(bands=2, momentum=0.5)),
    ("blt", blt_strategy(max_buffers=1, momentum=0.5)),
    ("bsr", bsr_strategy(bandwidth=2, alpha=0.9, beta=0.5)),
    ("bisr", bisr_strategy(bandwidth=2, momentum=0.2)),
    ("lambda_cgd", lambda_cgd_strategy(lambda_=0.5)),
]


def _template(backend_case) -> dict[str, object]:
    return {
        "weight": backend_case.array(
            np.zeros((2, 2)), dtype=backend_case.dtype("float32")
        ),
        "bias": backend_case.array(np.zeros((2,)), dtype=backend_case.dtype("float32")),
    }


def _clipped_grads(backend_case) -> object:
    return clipped(
        {
            "weight": backend_case.array(
                np.ones((2, 2)), dtype=backend_case.dtype("float32")
            ),
            "bias": backend_case.array(
                np.ones((2,)), dtype=backend_case.dtype("float32")
            ),
        },
        max_norm=0.25,
    )


def _assert_tree_equal(
    backend_case, left: dict[str, object], right: dict[str, object]
) -> None:
    for name in left:
        np.testing.assert_array_equal(
            backend_case.to_host(left[name]), backend_case.to_host(right[name])
        )


@pytest.mark.parametrize(("name", "strategy"), _STRATEGIES)
def test_mf_noise_strategies_run_on_portable_arrays(
    name: str, strategy: object, backend_case
) -> None:
    del name
    noise_fn, noise_state = mf_gaussian_noise(
        _template(backend_case),
        strategy,
        n_steps=_N_STEPS,
        noise_multiplier=0.5,
        key=key(41),
    )

    first, noise_state = noise_fn(_clipped_grads(backend_case), noise_state)
    second, noise_state = noise_fn(_clipped_grads(backend_case), noise_state)

    assert noise_state._step_counter == 2
    assert first.noise_stddev > 0.0
    assert second.noise_stddev > 0.0
    assert first.pytree["weight"].dtype == backend_case.dtype("float32")
    assert first.pytree["weight"].shape == (2, 2)
    assert not np.array_equal(
        backend_case.to_host(first.pytree["weight"]),
        backend_case.to_host(second.pytree["weight"]),
    )


@pytest.mark.parametrize(("name", "strategy"), _STRATEGIES)
def test_mf_noise_checkpoint_replays_the_next_mlx_step(
    name: str, strategy: object, backend_case
) -> None:
    del name
    noise_fn, noise_state = mf_gaussian_noise(
        _template(backend_case),
        strategy,
        n_steps=_N_STEPS,
        noise_multiplier=0.5,
        key=key(43),
    )
    _, saved_state = noise_fn(_clipped_grads(backend_case), noise_state)
    checkpoint = state_dict(saved_state)
    uninterrupted, uninterrupted_state = noise_fn(
        _clipped_grads(backend_case), saved_state
    )

    _, restore_template = mf_gaussian_noise(
        _template(backend_case),
        strategy,
        n_steps=_N_STEPS,
        noise_multiplier=0.5,
        key=key(43),
    )
    restored_state = from_state_dict(restore_template, checkpoint)
    resumed, resumed_state = noise_fn(_clipped_grads(backend_case), restored_state)

    _assert_tree_equal(backend_case, uninterrupted.pytree, resumed.pytree)
    assert uninterrupted_state._step_counter == resumed_state._step_counter == 2


def test_mf_private_second_moment_and_per_group_noise_run_portably(
    backend_case,
) -> None:
    template = _template(backend_case)
    strategy = band_mf_strategy(bands=2, momentum=0.5)
    noise_fn, noise_state = mf_gaussian_noise(
        template,
        strategy,
        n_steps=_N_STEPS,
        noise_multiplier=0.5,
        key=key(47),
        second_moment_strategy=band_mf_strategy(bands=2, momentum=0.25),
    )
    max_norm = PerGroup(
        groups={"weight": "weight", "bias": "bias"},
        values={"weight": 0.25, "bias": 0.5},
    )
    squared_max_norm = max_norm * max_norm
    grads = {
        name: backend_case.array(
            np.ones(backend_case.to_host(value).shape),
            dtype=backend_case.dtype("float32"),
        )
        for name, value in template.items()
    }
    paired = SecondMomentClippingOutput(
        grads=clipped(grads, max_norm=max_norm),
        squared_grads=clipped(
            {name: ops.multiply(value, value) for name, value in grads.items()},
            max_norm=squared_max_norm,
        ),
    )

    noised, noise_state = noise_fn(paired, noise_state)
    assert isinstance(noised, SecondMomentNoiseOutput)
    assert isinstance(noised.noisy_grads.noise_stddev, PerGroup)
    assert isinstance(noised.noisy_squared_grads.noise_stddev, PerGroup)
    assert noise_state._step_counter == 1


@pytest.mark.parametrize(
    "sampler_factory",
    [
        lambda: CyclicPoissonSampler(
            range(12),
            sample_rate=0.5,
            bands=1,
            n_steps=4,
            truncated_batch_size=3,
            key=key(53),
        ),
        lambda: CyclicPoissonSampler(
            range(12), sample_rate=0.5, bands=2, n_steps=4, key=key(57)
        ),
        lambda: BMinSepSampler(
            range(12), bands=2, sampling_prob=0.5, n_steps=4, key=key(59)
        ),
        lambda: BallsInBinsSampler(range(12), num_bins=2, n_steps=4, key=key(61)),
        lambda: SequentialBatchSampler(range(12), batch_size=3, n_steps=4),
    ],
)
def test_dpftrl_samplers_restore_the_remaining_stream(
    sampler_factory: Callable[[], object],
) -> None:
    sampler = sampler_factory()
    iterator = iter(sampler)
    next(iterator)
    next(iterator)
    checkpoint = state_dict(sampler)
    uninterrupted = list(iterator)

    restored = from_state_dict(sampler_factory(), checkpoint)
    assert list(restored) == uninterrupted
