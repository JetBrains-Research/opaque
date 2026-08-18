from collections.abc import Iterable

import pytest

from opaque.dpftrl.sampling import (
    BallsInBinsSampler,
    BMinSepSampler,
    CyclicPoissonSampler,
    SequentialBatchSampler,
)
from opaque.random import key
from opaque.sampling import Sampler


@pytest.mark.parametrize(
    "sampler",
    [
        BMinSepSampler(
            list(range(8)), bands=2, sampling_prob=0.5, n_steps=2, key=key(1)
        ),
        BallsInBinsSampler(list(range(8)), num_bins=2, n_steps=2, key=key(2)),
        CyclicPoissonSampler(list(range(8)), sample_rate=0.5, n_steps=2, key=key(3)),
        SequentialBatchSampler(list(range(8)), batch_size=4),
    ],
)
def test_sampler_uses_shared_contract(sampler: Sampler[list[int]]) -> None:
    batches = list(sampler)

    assert isinstance(sampler, Sampler)
    assert isinstance(sampler, Iterable)
    assert len(batches) == 2
    assert all(isinstance(batch, list) for batch in batches)
    assert all(isinstance(index, int) for batch in batches for index in batch)
