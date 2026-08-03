"""Global k-out-of-t sampler contracts."""

from __future__ import annotations

from collections import Counter

import pytest

from opaque.dpsgd.sampling import KOutOfTSampler
from opaque.random import key
from opaque.serialization import from_state_dict, state_dict


def test_each_record_participates_exactly_k_times():
    sampler = KOutOfTSampler(
        list(range(100)),
        total_participations=3,
        n_steps=10,
        key=key(7),
    )
    counts = Counter(index for batch in sampler for index in batch)
    assert set(counts.values()) == {3}


def test_stream_is_reproducible_and_resumable():
    def make(seed):
        return KOutOfTSampler(
            list(range(40)),
            total_participations=2,
            n_steps=8,
            key=key(seed),
        )

    expected = list(make(3))
    partial = make(3)
    iterator = iter(partial)
    for _ in range(5):
        next(iterator)
    restored = from_state_dict(make(99), state_dict(partial))

    assert restored.consumed == 5
    assert list(restored) == expected[5:]


def test_validation():
    with pytest.raises(ValueError, match="total_participations"):
        KOutOfTSampler([1], total_participations=3, n_steps=2, key=key(0))
    with pytest.raises(ValueError, match="n_steps"):
        KOutOfTSampler([1], total_participations=1, n_steps=0, key=key(0))
