"""Torch DataLoader compatibility for DP-FTRL batch samplers."""

from __future__ import annotations

from collections.abc import Iterable

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


def _identity(batch):
    return batch


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
def test_sampler_is_accepted_as_a_dataloader_batch_sampler(
    sampler: Iterable[list[int]],
) -> None:
    dataset = TensorDataset(torch.arange(8))
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=_identity)

    batches = list(loader)

    assert isinstance(sampler, Sampler)
    assert isinstance(sampler, Iterable)
    assert len(batches) == 2
    assert all(isinstance(batch, list) for batch in batches)


def test_cyclic_poisson_dataloader_integration() -> None:
    dataset = list(range(100))
    sampler = CyclicPoissonSampler(
        dataset,
        sample_rate=0.5,
        bands=2,
        n_steps=10,
        key=key(42),
    )

    batches = list(DataLoader(dataset, batch_sampler=sampler))

    assert len(batches) == 10
    assert all(isinstance(batch, torch.Tensor) for batch in batches)
    assert all(0 <= item < 100 for batch in batches for item in batch)


def test_cyclic_poisson_dataloader_handles_variable_batch_sizes() -> None:
    dataset = list(range(200))
    sampler = CyclicPoissonSampler(
        dataset,
        sample_rate=0.3,
        bands=1,
        n_steps=20,
        key=key(42),
    )

    batch_sizes = [len(batch) for batch in DataLoader(dataset, batch_sampler=sampler)]

    assert len(set(batch_sizes)) > 1


def test_cyclic_poisson_accepts_torch_dataset() -> None:
    dataset = TensorDataset(torch.arange(100))
    sampler = CyclicPoissonSampler(
        dataset,
        sample_rate=0.5,
        bands=3,
        n_steps=9,
        key=key(42),
    )

    batches = list(sampler)

    assert len(batches) == 9
    assert all(0 <= index < 100 for batch in batches for index in batch)


def test_sequential_dataloader_integration() -> None:
    data = torch.arange(50).unsqueeze(1).float()
    dataset = TensorDataset(data)
    sampler = SequentialBatchSampler(dataset, batch_size=10)

    batches = list(DataLoader(dataset, batch_sampler=sampler))

    assert len(batches) == 5
    assert torch.equal(batches[0][0], torch.arange(10).unsqueeze(1).float())
