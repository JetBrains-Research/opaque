"""``state_dict`` / ``from_state_dict`` round-trip for :class:`PoissonSampler`.

Verifies the registry-based serialization pair behaves correctly:

* A snapshot taken mid-iteration, restored into a freshly-built sampler,
  yields the same remaining batches that the original would have yielded.
* ``consumed`` is honoured on the restored sampler so its iterator
  starts at the saved cursor.
* The ``data_source`` is supplied by the ``template`` (not serialised).
"""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from opaque.dpsgd.sampling import PoissonSampler
from opaque.random import key
from opaque.serialization import state_dict, from_state_dict


def _make_sampler(seed: int = 7, n_steps: int = 20) -> PoissonSampler:
    dataset = TensorDataset(torch.arange(200).reshape(-1, 1))
    return PoissonSampler(
        dataset, sample_rate=0.1, n_steps=n_steps, key=key(seed)
    )


class TestPoissonStateDictRoundTrip:
    def test_snapshot_at_step_k_resumes_with_matching_remainder(self):
        """Snapshot mid-iter → load → remaining yields match the original tail."""
        original = _make_sampler(seed=42)
        original_batches = list(original)

        # Build a fresh sampler, iterate up to step K, snapshot.
        fresh = _make_sampler(seed=42)
        it = iter(fresh)
        K = 7
        first_k = [next(it) for _ in range(K)]
        assert first_k == original_batches[:K]
        snapshot = state_dict(fresh)

        # Load into a NEW template (different RNG to prove the snapshot,
        # not the template's key, drives the restored state).
        template = _make_sampler(seed=99)
        restored = from_state_dict(template, snapshot)

        assert restored.consumed == K
        assert list(restored) == original_batches[K:]

    def test_snapshot_at_zero_is_full_run(self):
        original = _make_sampler(seed=11)
        original_batches = list(original)

        fresh = _make_sampler(seed=11)
        snapshot = state_dict(fresh)
        template = _make_sampler(seed=0)  # different seed; template only supplies dataset
        restored = from_state_dict(template, snapshot)

        assert restored.consumed == 0
        assert list(restored) == original_batches

    def test_snapshot_at_completion_yields_empty(self):
        sampler = _make_sampler(seed=1, n_steps=5)
        _ = list(sampler)
        assert sampler.consumed == 5

        snapshot = state_dict(sampler)
        template = _make_sampler(seed=999, n_steps=5)
        restored = from_state_dict(template, snapshot)

        assert restored.consumed == 5
        assert list(restored) == []

    def test_round_trip_preserves_truncated_batch_size(self):
        dataset = TensorDataset(torch.arange(300).reshape(-1, 1))
        sampler = PoissonSampler(
            dataset,
            sample_rate=0.2,
            n_steps=10,
            truncated_batch_size=15,
            key=key(7),
        )
        snapshot = state_dict(sampler)
        template = PoissonSampler(
            dataset, sample_rate=0.5, n_steps=99, key=key(0)
        )
        restored = from_state_dict(template, snapshot)

        assert restored.sample_rate == 0.2
        assert restored.n_steps == 10
        assert restored.truncated_batch_size == 15
