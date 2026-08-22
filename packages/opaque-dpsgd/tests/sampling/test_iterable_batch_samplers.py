"""Samplers expose the provider-neutral iterable batch-sampler contract."""

from collections.abc import Iterable

import pytest

from opaque.dpsgd.sampling import (
    KOutOfTSampler,
    PoissonSampler,
    RandomAllocationSampler,
)
from opaque.random import key
from opaque.sampling import Sampler


@pytest.mark.parametrize(
    "sampler",
    [
        PoissonSampler(list(range(8)), sample_rate=0.5, n_steps=2, key=key(1)),
        KOutOfTSampler(list(range(8)), total_participations=1, n_steps=2, key=key(2)),
        RandomAllocationSampler(list(range(8)), num_bins=2, n_steps=2, key=key(3)),
    ],
)
def test_sampler_is_iterable_and_yields_index_lists(
    sampler: Iterable[list[int]],
) -> None:
    assert isinstance(sampler, Sampler)
    assert isinstance(sampler, Iterable)
    assert all(isinstance(batch, list) for batch in sampler)
