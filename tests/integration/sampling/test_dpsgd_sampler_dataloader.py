"""DP-SGD samplers work as structural Torch ``DataLoader`` batch samplers."""

from __future__ import annotations

import pytest
import torch

from opaque.dpsgd.sampling import (
    KOutOfTSampler,
    PoissonSampler,
    RandomAllocationSampler,
)
from opaque.random import key


@pytest.mark.parametrize(
    ("sampler_type", "kwargs"),
    [
        (PoissonSampler, {"sample_rate": 1.0, "n_steps": 2}),
        (KOutOfTSampler, {"total_participations": 2, "n_steps": 2}),
        (RandomAllocationSampler, {"num_bins": 2, "n_steps": 2}),
    ],
    ids=("poisson", "k-out-of-t", "random-allocation"),
)
def test_dpsgd_sampler_is_a_torch_batch_sampler(sampler_type, kwargs) -> None:
    dataset = list(range(100))
    sampler = sampler_type(dataset, key=key(13), **kwargs)
    expected = list(sampler_type(dataset, key=key(13), **kwargs))

    loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=sampler, collate_fn=list
    )

    assert list(loader) == expected
