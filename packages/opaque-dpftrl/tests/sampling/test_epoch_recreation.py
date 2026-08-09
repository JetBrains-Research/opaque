"""Regression tests for per-epoch sampler recreation (issue #357).

The DP-FTRL example builds one DataLoader per epoch from
``make_epoch_sampler(epoch)``.  Samplers are single-pass (the
``_consumed`` cursor persists across ``__iter__`` calls), so a factory
returning a cached sampler silently yields zero batches from epoch 2
onwards.  These tests pin the contract the fixed example relies on:
reconstructing a sampler with the same arguments (and, for BnB, the
same key) reproduces the identical batch stream, while reusing the
exhausted object yields nothing.
"""

import torch
from torch.utils.data import TensorDataset

from opaque.dpftrl.sampling import BallsInBinsSampler, SequentialBatchSampler
from opaque.random import key


def _ds(n=64):
    return TensorDataset(torch.arange(n).unsqueeze(1).float())


class TestExhaustedSamplerReuse:
    """Reusing one sampler across epochs trains only the first epoch."""

    def test_sequential_reuse_yields_zero_batches(self):
        sampler = SequentialBatchSampler(_ds(64), batch_size=8)
        assert len(list(sampler)) == 8
        assert list(sampler) == []  # exhausted: the cached-factory bug
        assert len(sampler) == 0

    def test_balls_in_bins_reuse_yields_zero_batches(self):
        sampler = BallsInBinsSampler(_ds(64), num_bins=8, n_steps=8, key=key(7))
        assert len(list(sampler)) == 8
        assert list(sampler) == []
        assert len(sampler) == 0


class TestFreshPerEpochFactory:
    """The fixed pattern: construct a fresh sampler every epoch."""

    def test_sequential_fresh_per_epoch_yields_full_epochs(self):
        num_epochs, steps_per_epoch = 2, 8
        ds = _ds(64)

        def make_epoch_sampler(epoch):
            return SequentialBatchSampler(ds, batch_size=8)

        per_epoch = [list(make_epoch_sampler(e)) for e in range(num_epochs)]
        assert [len(b) for b in per_epoch] == [steps_per_epoch] * num_epochs
        # Deterministic: every epoch repeats the identical fixed order
        # (the min_sep / max_participations contract for BLT).
        assert per_epoch[0] == per_epoch[1]

    def test_balls_in_bins_same_key_reproduces_partition(self):
        num_epochs, steps_per_epoch = 2, 8
        ds = _ds(64)

        def make_epoch_sampler(epoch):
            # SAME key every epoch: the bin assignment is deterministic
            # from the key, preserving the fixed partition required by
            # BnB accounting (Lemma 3.2, Choquette-Choo et al. 2024).
            return BallsInBinsSampler(
                ds, num_bins=steps_per_epoch, n_steps=steps_per_epoch, key=key(7)
            )

        per_epoch = [list(make_epoch_sampler(e)) for e in range(num_epochs)]
        assert [len(b) for b in per_epoch] == [steps_per_epoch] * num_epochs
        assert per_epoch[0] == per_epoch[1]

    def test_two_epoch_total_step_count(self):
        # 2 epochs must produce 2 x steps_per_epoch batches; the
        # exhausted-reuse bug collapses this to 1 x steps_per_epoch.
        ds = _ds(64)
        total = sum(
            len(list(SequentialBatchSampler(ds, batch_size=8))) for _ in range(2)
        )
        assert total == 16
