"""RandomAllocationSampler: partition, redraw, empty batches, resume."""

from __future__ import annotations

from itertools import chain

import pytest

from opaque.dpsgd.sampling import RandomAllocationSampler
from opaque.random import key
from opaque.serialization import from_state_dict, state_dict

_N = 200
_BINS = 8


def _ds(n: int = _N) -> list[int]:
    return list(range(n))


def _make(n: int = _N, num_bins: int = _BINS, n_steps: int | None = 32, seed: int = 7):
    return RandomAllocationSampler(_ds(n), num_bins, n_steps, key=key(seed))


class TestAllocationStructure:
    def test_each_epoch_partitions_the_dataset(self):
        """The defining property: every epoch's bins partition the dataset
        exactly — no example missing, none twice."""
        s = _make()
        batches = list(s)
        for e in range(len(batches) // _BINS):
            epoch = batches[e * _BINS : (e + 1) * _BINS]
            assert sorted(chain(*epoch)) == list(range(_N)), (
                f"epoch {e} not a partition"
            )

    def test_yields_exactly_n_steps(self):
        s = _make(n_steps=32)
        assert len(list(s)) == 32

    def test_redraws_every_epoch(self):
        """The assignment must differ across epochs — that is the whole
        difference from DP-FTRL's fixed-assignment balls-in-bins."""
        s = _make(n_steps=_BINS * 4)
        batches = list(s)
        epochs = [batches[e * _BINS : (e + 1) * _BINS] for e in range(4)]
        assert any(epochs[0] != epochs[e] for e in range(1, 4)), (
            "assignment did not change across epochs"
        )

    def test_emits_empty_batches(self):
        """Bin sizes are Binomial, so empty bins occur and must be emitted
        rather than compacted away."""
        s = RandomAllocationSampler(_ds(3), 8, 8, key=key(1))
        batches = list(s)
        assert len(batches) == 8, "must emit one batch per bin, including empties"
        assert any(b == [] for b in batches), "expected at least one empty batch"
        assert sorted(chain(*batches)) == [0, 1, 2]

    def test_expected_batch_size(self):
        s = _make()
        assert s.expected_batch_size == pytest.approx(_N / _BINS)

    def test_num_epochs(self):
        assert _make(n_steps=32).num_epochs == 4
        assert _make(n_steps=12).num_epochs == 2
        assert _make(n_steps=None).num_epochs is None


class TestValidation:
    def test_rejects_empty_dataset(self):
        with pytest.raises(ValueError, match="must not be empty"):
            RandomAllocationSampler([], 4, 4, key=key(0))

    def test_rejects_num_bins_below_two(self):
        with pytest.raises(ValueError, match="num_bins"):
            RandomAllocationSampler(_ds(), 1, 4, key=key(0))

    def test_allows_partial_final_epoch(self):
        sampler = RandomAllocationSampler(_ds(), 8, 12, key=key(0))

        batches = list(sampler)

        assert len(batches) == 12
        assert sorted(chain(*batches[:8])) == list(range(_N))
        assert len(batches[8:]) == 4

    def test_requires_keyword_only_key(self):
        with pytest.raises(TypeError):
            RandomAllocationSampler(_ds(), 8, 8)  # type: ignore[call-arg]

    def test_len_raises_when_unbounded(self):
        s = _make(n_steps=None)
        with pytest.raises(TypeError, match="unsized"):
            len(s)


class TestReproducibility:
    def test_same_key_same_stream(self):
        assert list(_make(seed=99)) == list(_make(seed=99))

    def test_different_key_different_stream(self):
        assert list(_make(seed=1)) != list(_make(seed=2))


class TestLenReflectsRemaining:
    def test_len_decreases_as_consumed(self):
        s = _make(n_steps=32)
        assert len(s) == 32
        it = iter(s)
        for _ in range(11):
            next(it)
        assert len(s) == 32 - 11


class TestStateDict:
    def test_snapshot_resumes_at_cursor(self):
        fresh = _make(seed=5)
        original = list(_make(seed=5))

        it = iter(fresh)
        k = 11
        for _ in range(k):
            next(it)
        snapshot = state_dict(fresh)

        # Restore into a template built with a *different* key, to prove the
        # snapshot rather than the template drives the restored stream.
        restored = from_state_dict(_make(seed=99), snapshot)
        assert restored.consumed == k
        assert list(restored) == original[k:]

    def test_resume_at_epoch_boundary(self):
        fresh = _make(seed=5)
        original = list(_make(seed=5))
        it = iter(fresh)
        for _ in range(_BINS):  # exactly one epoch
            next(it)
        restored = from_state_dict(_make(seed=99), state_dict(fresh))
        assert restored.consumed == _BINS
        assert list(restored) == original[_BINS:]

    def test_snapshot_at_zero_is_full_run(self):
        fresh = _make(seed=5)
        restored = from_state_dict(_make(seed=99), state_dict(fresh))
        assert restored.consumed == 0
        assert list(restored) == list(_make(seed=5))

    def test_snapshot_at_completion_yields_empty(self):
        fresh = _make(seed=5)
        list(fresh)
        restored = from_state_dict(_make(seed=99), state_dict(fresh))
        assert restored.consumed == 32
        assert list(restored) == []

    def test_num_bins_comes_from_snapshot_n_steps_from_template(self):
        """``num_bins`` is the amplification factor, so it must survive the
        round trip; ``n_steps`` follows the template so a run can be extended."""
        fresh = RandomAllocationSampler(_ds(), 8, 16, key=key(3))
        snapshot = state_dict(fresh)
        template = RandomAllocationSampler(_ds(), 4, 40, key=key(77))
        restored = from_state_dict(template, snapshot)
        assert restored.num_bins == 8
        assert restored.n_steps == 40

    def test_rejects_dataset_length_mismatch(self):
        fresh = _make(seed=5)
        snapshot = state_dict(fresh)
        template = _make(n=_N + 1, seed=5)
        with pytest.raises(ValueError, match="num_samples"):
            from_state_dict(template, snapshot)
