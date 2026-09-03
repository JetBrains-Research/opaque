"""Torch data-loader compatibility for DP-SGD iterable batch samplers."""

from __future__ import annotations

from collections.abc import Iterable

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from opaque.dpsgd.sampling import (
    KOutOfTSampler,
    PoissonSampler,
    RandomAllocationSampler,
)
from opaque.random import key
from opaque.sampling import Sampler


def _identity(batch):
    return batch


@pytest.mark.parametrize(
    "sampler",
    [
        PoissonSampler(list(range(16)), sample_rate=0.75, n_steps=2, key=key(1)),
        KOutOfTSampler(list(range(16)), total_participations=1, n_steps=2, key=key(2)),
        RandomAllocationSampler(list(range(16)), num_bins=2, n_steps=2, key=key(3)),
    ],
)
def test_sampler_is_accepted_as_a_dataloader_batch_sampler(
    sampler: Iterable[list[int]],
) -> None:
    dataset = TensorDataset(torch.arange(16))
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=_identity)

    batches = list(loader)

    assert isinstance(sampler, Sampler)
    assert isinstance(sampler, Iterable)
    assert len(batches) == 2
    assert all(isinstance(batch, list) for batch in batches)


def test_poisson_default_collation_yields_variable_stacked_batches() -> None:
    # Variable-size Poisson batches must survive torch's default collation:
    # each yielded batch stacks to (batch, features) with the Poisson-drawn
    # batch dimension varying across steps.
    dataset = TensorDataset(torch.randn(1000, 10), torch.randn(1000, 5))
    sampler = PoissonSampler(dataset, sample_rate=0.1, n_steps=5, key=key(42))

    loader = DataLoader(dataset, batch_sampler=sampler)

    batch_sizes = []
    for features, targets in loader:
        assert features.shape[1] == 10
        assert targets.shape[1] == 5
        batch_sizes.append(features.shape[0])

    assert len(batch_sizes) == 5
    assert len(set(batch_sizes)) > 1
