import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

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
    dataset = TensorDataset(torch.arange(8))
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=lambda batch: batch)

    assert isinstance(sampler, Sampler)
    assert all(isinstance(batch, list) for batch in loader)
