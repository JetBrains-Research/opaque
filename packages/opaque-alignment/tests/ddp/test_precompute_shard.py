"""CPU gloo regressions: reference precompute must shard across ranks."""

from __future__ import annotations

import tempfile

import pytest
import torch.distributed as dist
from alignment_ddp_helpers import (
    _spawn_gloo,
    _worker_divergent_cache_state,
    _worker_divergent_use_cache,
    _worker_empty_shard_preserves_dtype,
    _worker_mismatched_dataset_sizes,
    _worker_shards_and_restores_order,
    _worker_shared_cache_hit,
)


def _require_gloo() -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")
    if not dist.is_gloo_available():
        pytest.skip("gloo backend is not available")


@pytest.mark.parametrize(
    ("world_size", "n_rows"),
    [
        (2, 6),  # even split
        (2, 7),  # remainder goes to the last rank
        (3, 2),  # fewer rows than ranks: the leading ranks get empty shards
    ],
)
def test_each_rank_scores_only_its_shard(world_size: int, n_rows: int) -> None:
    """Every split reproduces the single-process columns in dataset order."""
    _require_gloo()
    with tempfile.TemporaryDirectory() as tmp:
        _spawn_gloo(world_size, _worker_shards_and_restores_order, n_rows, tmp)


def test_empty_shard_does_not_dictate_gathered_dtype() -> None:
    _require_gloo()
    with tempfile.TemporaryDirectory() as tmp:
        _spawn_gloo(2, _worker_empty_shard_preserves_dtype, tmp)


def test_cache_visible_to_every_rank_is_reused() -> None:
    _require_gloo()
    with tempfile.TemporaryDirectory() as tmp:
        _spawn_gloo(2, _worker_shared_cache_hit, 6, tmp)


def test_cache_visible_to_one_rank_makes_the_group_recompute() -> None:
    _require_gloo()
    with tempfile.TemporaryDirectory() as tmp:
        _spawn_gloo(2, _worker_divergent_cache_state, 6, tmp)


def test_ranks_disagreeing_on_use_cache_do_not_deadlock() -> None:
    _require_gloo()
    with tempfile.TemporaryDirectory() as tmp:
        _spawn_gloo(2, _worker_divergent_use_cache, 6, tmp)


def test_ranks_holding_different_datasets_are_rejected() -> None:
    _require_gloo()
    _spawn_gloo(2, _worker_mismatched_dataset_sizes)
