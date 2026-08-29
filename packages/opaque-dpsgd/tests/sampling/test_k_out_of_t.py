"""Block and total k-out-of-t sampler contracts."""

from __future__ import annotations

from collections import Counter
from itertools import chain

import pytest

from opaque.dpsgd.sampling import KOutOfTSampler
from opaque.random import key
from opaque.serialization import from_state_dict, state_dict


def test_total_allocation_selects_each_record_exactly_k_times():
    sampler = KOutOfTSampler(
        list(range(100)),
        k=3,
        t=10,
        allocation="total",
        key=key(7),
    )
    counts = Counter(index for batch in sampler for index in batch)
    assert set(counts.values()) == {3}


def test_block_allocation_partitions_every_block():
    sampler = KOutOfTSampler(
        list(range(100)),
        k=4,
        t=19,
        allocation="block",
        key=key(7),
    )
    batches = list(sampler)
    offset = 0

    assert sampler.block_sizes == (4, 5, 5, 5)
    for block_size in sampler.block_sizes:
        assert sorted(chain(*batches[offset : offset + block_size])) == list(range(100))
        offset += block_size


def test_block_allocation_draws_each_block_independently():
    sampler = KOutOfTSampler(
        list(range(100)),
        k=4,
        t=20,
        allocation="block",
        key=key(7),
    )
    batches = list(sampler)
    epochs = [batches[offset : offset + 5] for offset in range(0, 20, 5)]

    assert any(epoch != epochs[0] for epoch in epochs[1:])


def test_allocation_modes_use_distinct_stream_domains():
    kwargs = {"data_source": list(range(100)), "k": 4, "t": 20, "key": key(7)}

    block = list(KOutOfTSampler(allocation="block", **kwargs))
    total = list(KOutOfTSampler(allocation="total", **kwargs))

    assert block != total


@pytest.mark.parametrize("allocation", ["block", "total"])
def test_stream_is_reproducible_and_resumable(allocation: str):
    def make(seed: int):
        return KOutOfTSampler(
            list(range(40)),
            k=2,
            t=8,
            allocation=allocation,  # type: ignore[arg-type]
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


def test_expected_batch_size_and_block_sizes():
    block = KOutOfTSampler(
        list(range(100)),
        k=4,
        t=19,
        allocation="block",
        key=key(0),
    )
    total = KOutOfTSampler(
        list(range(100)),
        k=4,
        t=19,
        allocation="total",
        key=key(0),
    )

    assert block.expected_batch_size == pytest.approx(100 * 4 / 19)
    assert total.expected_batch_size == pytest.approx(100 * 4 / 19)
    assert block.block_sizes == (4, 5, 5, 5)
    assert total.block_sizes is None


def test_validation():
    with pytest.raises(ValueError, match="k"):
        KOutOfTSampler([1], k=3, t=2, allocation="total", key=key(0))
    with pytest.raises(ValueError, match="t"):
        KOutOfTSampler([1], k=1, t=0, allocation="total", key=key(0))
    with pytest.raises(ValueError, match="allocation"):
        KOutOfTSampler([1], k=1, t=1, allocation="bad", key=key(0))  # type: ignore[arg-type]
