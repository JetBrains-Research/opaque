from __future__ import annotations

from opaque.dpftrl.sampling import BallsInBinsSampler
from opaque.random import key


def test_balls_in_bins_covers_dataset_and_repeats_assignment():
    n_samples, num_bins, num_epochs = 1000, 10, 3
    sampler = BallsInBinsSampler(
        list(range(n_samples)),
        num_bins=num_bins,
        n_steps=num_bins * num_epochs,
        key=key(42),
    )
    batches = list(sampler)
    first_epoch = batches[:num_bins]
    assert {index for batch in first_epoch for index in batch} == set(range(n_samples))
    assert sum(map(len, first_epoch)) == n_samples
    for epoch in range(1, num_epochs):
        assert batches[epoch * num_bins : (epoch + 1) * num_bins] == first_epoch


def test_balls_in_bins_has_variable_bin_sizes():
    sampler = BallsInBinsSampler(
        list(range(10000)), num_bins=50, n_steps=50, key=key(123)
    )
    assert len({len(batch) for batch in sampler}) > 1
