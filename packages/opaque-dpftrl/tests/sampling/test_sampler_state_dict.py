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
    BallsInBinsSampler,
    BMinSepSampler,
    CyclicPoissonSampler,
    SequentialBatchSampler,
)
from opaque.random import key
from opaque.serialization import from_state_dict, state_dict


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

        # Sample-math args (sample_rate, bands, truncated_batch_size,
        # key) come from the snapshot.
        assert restored.sample_rate == 0.15
        assert restored.bands == 5
        assert restored.truncated_batch_size == 8
        # ``n_steps`` follows the template (allows resume + extend).
        assert restored.n_steps == 10


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
    def test_emits_empty_bin_slots(self):
        sampler = BallsInBinsSampler(_ds(1), num_bins=2, n_steps=4, key=key(5))

        batches = list(sampler)

        assert len(batches) == 4
        assert batches[:2] == batches[2:]
        assert sorted(index for batch in batches[:2] for index in batch) == [0]
        assert any(not batch for batch in batches[:2])

    def test_snapshot_resumes_at_cursor(self):
        original = BallsInBinsSampler(_ds(), num_bins=10, n_steps=30, key=key(5))
        original_batches = list(original)

        fresh = BallsInBinsSampler(_ds(), num_bins=10, n_steps=30, key=key(5))
        it = iter(fresh)
        K = 12
        head = [next(it) for _ in range(K)]
        assert head == original_batches[:K]

        snapshot = state_dict(fresh)
        template = BallsInBinsSampler(_ds(), num_bins=10, n_steps=30, key=key(99))
        restored = from_state_dict(template, snapshot)

        assert restored.consumed == K
        assert list(restored) == original_batches[K:]

    def test_constructor_args_preserved(self):
        sampler = BallsInBinsSampler(_ds(), num_bins=8, n_steps=16, key=key(1))
        snapshot = state_dict(sampler)
        # ``num_bins`` drives the bin assignment — must match across
        # save/resume to preserve the BnB privacy contract.  Use the
        # same value on the template so the constructor's
        # ``n_steps % num_bins == 0`` check passes; ``n_steps`` itself
        # is the per-run iteration bound and comes from the template.
        template = BallsInBinsSampler(_ds(), num_bins=8, n_steps=24, key=key(0))
        restored = from_state_dict(template, snapshot)

        assert restored.num_bins == 8
        assert restored.n_steps == 24


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

    def test_cycling_snapshot_resumes_past_first_pass(self):
        """A cursor beyond one pass round-trips (``n_steps`` cycles);
        the template's ``n_steps`` fixes the restored horizon."""
        full = list(SequentialBatchSampler(_ds(200), batch_size=10, n_steps=60))

        fresh = SequentialBatchSampler(_ds(200), batch_size=10, n_steps=60)
        it = iter(fresh)
        K = 27  # past the first 20-batch pass
        head = [next(it) for _ in range(K)]
        assert head == full[:K]

        snapshot = state_dict(fresh)
        template = SequentialBatchSampler(_ds(200), batch_size=10, n_steps=60)
        restored = from_state_dict(template, snapshot)

        assert restored.consumed == K
        assert len(restored) == 60 - K
        assert list(restored) == full[K:]


class TestLenReflectsRemaining:
    """``__len__`` reports remaining batches, not the original total."""

    def test_cyclic_poisson_len_drops(self):
        s = CyclicPoissonSampler(
            _ds(), sample_rate=0.1, bands=4, n_steps=20, key=key(1)
        )
        assert len(s) == 20
        it = iter(s)
        for _ in range(8):
            next(it)
        assert len(s) == 12

    def test_b_min_sep_len_drops(self):
        s = BMinSepSampler(_ds(), bands=3, sampling_prob=0.2, n_steps=12, key=key(1))
        assert len(s) == 12
        it = iter(s)
        for _ in range(5):
            next(it)
        assert len(s) == 7

    def test_balls_in_bins_len_drops(self):
        s = BallsInBinsSampler(_ds(), num_bins=10, n_steps=30, key=key(5))
        total = len(s)
        it = iter(s)
        for _ in range(7):
            next(it)
        assert len(s) == total - 7

    def test_sequential_len_drops(self):
        s = SequentialBatchSampler(_ds(200), batch_size=10)
        assert len(s) == 20
        it = iter(s)
        for _ in range(6):
            next(it)
        assert len(s) == 14


class TestRejectDatasetLengthMismatch:
    """All four ``from_state_dict`` paths validate template dataset length."""

    def test_cyclic_poisson_mismatch_raises(self):
        import pytest

        sampler = CyclicPoissonSampler(
            _ds(200), sample_rate=0.1, bands=4, n_steps=20, key=key(7)
        )
        snapshot = state_dict(sampler)
        template = CyclicPoissonSampler(
            _ds(150), sample_rate=0.1, bands=4, n_steps=20, key=key(0)
        )
        with pytest.raises(ValueError, match="num_examples"):
            from_state_dict(template, snapshot)

    def test_b_min_sep_mismatch_raises(self):
        import pytest

        sampler = BMinSepSampler(
            _ds(200), bands=3, sampling_prob=0.2, n_steps=12, key=key(7)
        )
        snapshot = state_dict(sampler)
        template = BMinSepSampler(
            _ds(150), bands=3, sampling_prob=0.2, n_steps=12, key=key(0)
        )
        with pytest.raises(ValueError, match="num_examples"):
            from_state_dict(template, snapshot)

    def test_balls_in_bins_mismatch_raises(self):
        import pytest

        sampler = BallsInBinsSampler(_ds(200), num_bins=10, n_steps=30, key=key(7))
        snapshot = state_dict(sampler)
        template = BallsInBinsSampler(_ds(150), num_bins=10, n_steps=30, key=key(0))
        with pytest.raises(ValueError, match="num_samples"):
            from_state_dict(template, snapshot)

    def test_sequential_mismatch_raises(self):
        import pytest

        sampler = SequentialBatchSampler(_ds(200), batch_size=10)
        snapshot = state_dict(sampler)
        template = SequentialBatchSampler(_ds(150), batch_size=10)
        with pytest.raises(ValueError, match="num_samples"):
            from_state_dict(template, snapshot)
