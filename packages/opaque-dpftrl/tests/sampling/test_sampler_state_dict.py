"""``state_dict`` / ``from_state_dict`` round-trip for DP-FTRL samplers.

Covers all four dp-ftrl samplers:

* :class:`CyclicPoissonSampler` — partition deterministic from key;
  generator replay matches a continuous run.
* :class:`BMinSepSampler` — Markov state (cooldown array + recent deque)
  must be reconstructed by replay; round-trip preserves the b-min-sep
  invariant.
* :class:`BallsInBinsSampler` — bin assignment deterministic; cursor
  restoration alone fixes resume.
* :class:`SequentialBatchSampler` — pure deterministic; cursor only.
"""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from opaque.dpftrl.sampling import (
    BMinSepSampler,
    BallsInBinsSampler,
    CyclicPoissonSampler,
    SequentialBatchSampler,
)
from opaque.random import key
from opaque.serialization import state_dict, from_state_dict


def _ds(n: int = 200) -> TensorDataset:
    return TensorDataset(torch.arange(n).reshape(-1, 1))


class TestCyclicPoissonRoundTrip:
    def test_snapshot_resumes_with_matching_remainder(self):
        original = CyclicPoissonSampler(
            _ds(), sample_rate=0.1, bands=4, n_steps=20, key=key(42)
        )
        original_batches = list(original)

        fresh = CyclicPoissonSampler(
            _ds(), sample_rate=0.1, bands=4, n_steps=20, key=key(42)
        )
        it = iter(fresh)
        K = 9
        head = [next(it) for _ in range(K)]
        assert head == original_batches[:K]

        snapshot = state_dict(fresh)
        # Template with different key proves the snapshot owns RNG state.
        template = CyclicPoissonSampler(
            _ds(), sample_rate=0.1, bands=4, n_steps=20, key=key(0)
        )
        restored = from_state_dict(template, snapshot)

        assert restored.consumed == K
        assert list(restored) == original_batches[K:]

    def test_constructor_args_preserved(self):
        sampler = CyclicPoissonSampler(
            _ds(),
            sample_rate=0.15,
            bands=5,
            n_steps=25,
            truncated_batch_size=8,
            key=key(7),
        )
        snapshot = state_dict(sampler)
        template = CyclicPoissonSampler(
            _ds(), sample_rate=0.5, bands=2, n_steps=10, key=key(0)
        )
        restored = from_state_dict(template, snapshot)

        assert restored.sample_rate == 0.15
        assert restored.bands == 5
        assert restored.n_steps == 25
        assert restored.truncated_batch_size == 8


class TestBMinSepRoundTrip:
    def test_snapshot_replays_markov_state(self):
        """The cooldown array + recent deque must reconstruct under replay."""
        original = BMinSepSampler(
            _ds(), bands=3, sampling_prob=0.2, n_steps=12, key=key(13)
        )
        original_batches = list(original)

        fresh = BMinSepSampler(
            _ds(), bands=3, sampling_prob=0.2, n_steps=12, key=key(13)
        )
        it = iter(fresh)
        K = 5
        head = [next(it) for _ in range(K)]
        assert head == original_batches[:K]

        snapshot = state_dict(fresh)
        template = BMinSepSampler(
            _ds(), bands=3, sampling_prob=0.2, n_steps=12, key=key(0)
        )
        restored = from_state_dict(template, snapshot)

        # The Markov state (cooldown + recent deque) must match what a
        # continuous run would have at step K — verified by the fact
        # that the remaining batches match exactly.
        assert restored.consumed == K
        assert list(restored) == original_batches[K:]


class TestBallsInBinsRoundTrip:
    def test_snapshot_resumes_at_cursor(self):
        original = BallsInBinsSampler(
            _ds(), num_bins=10, n_steps=30, key=key(5)
        )
        original_batches = list(original)

        fresh = BallsInBinsSampler(
            _ds(), num_bins=10, n_steps=30, key=key(5)
        )
        it = iter(fresh)
        K = 12
        head = [next(it) for _ in range(K)]
        assert head == original_batches[:K]

        snapshot = state_dict(fresh)
        template = BallsInBinsSampler(
            _ds(), num_bins=10, n_steps=30, key=key(99)
        )
        restored = from_state_dict(template, snapshot)

        assert restored.consumed == K
        assert list(restored) == original_batches[K:]

    def test_constructor_args_preserved(self):
        sampler = BallsInBinsSampler(_ds(), num_bins=8, n_steps=16, key=key(1))
        snapshot = state_dict(sampler)
        template = BallsInBinsSampler(_ds(), num_bins=2, n_steps=4, key=key(0))
        restored = from_state_dict(template, snapshot)

        assert restored.num_bins == 8
        assert restored.n_steps == 16


class TestSequentialRoundTrip:
    def test_snapshot_resumes_at_cursor(self):
        sampler = SequentialBatchSampler(_ds(200), batch_size=10)
        original_batches = list(sampler)

        fresh = SequentialBatchSampler(_ds(200), batch_size=10)
        it = iter(fresh)
        K = 8
        head = [next(it) for _ in range(K)]
        assert head == original_batches[:K]

        snapshot = state_dict(fresh)
        template = SequentialBatchSampler(_ds(200), batch_size=10)
        restored = from_state_dict(template, snapshot)

        assert restored.consumed == K
        assert list(restored) == original_batches[K:]

    def test_constructor_args_preserved(self):
        sampler = SequentialBatchSampler(_ds(200), batch_size=25)
        snapshot = state_dict(sampler)
        template = SequentialBatchSampler(_ds(200), batch_size=100)
        restored = from_state_dict(template, snapshot)

        assert restored._batch_size == 25
