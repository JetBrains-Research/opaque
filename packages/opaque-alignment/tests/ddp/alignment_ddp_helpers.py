"""Gloo CPU workers for the sharded reference-logprob precompute.

Workers must live in a top-level-importable module because ``mp.spawn``
pickles the target by qualified name; ``tests/conftest.py`` puts this
directory on ``sys.path`` and ``PYTHONPATH`` for the parent and the children.
"""

from __future__ import annotations

import os
import socket
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Short timeout: a mismatched collective sequence must fail the test promptly
# rather than hang the CI job.
_PG_TIMEOUT = timedelta(seconds=120)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _setup_gloo(rank: int, world_size: int, port: int) -> None:
    """CPU process-group init for the sharded-precompute regressions."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(
        backend="gloo", rank=rank, world_size=world_size, timeout=_PG_TIMEOUT
    )


def _cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _spawn_gloo(world_size: int, fn: Any, *args: Any) -> None:
    port = _find_free_port()
    mp.spawn(fn, args=(world_size, port, *args), nprocs=world_size, join=True)


# --------------------------------------------------------------------------
# Shared fixtures: a dataset, a collator, and a ref that records what it saw.
# --------------------------------------------------------------------------


def make_dataset(n_rows: int, start: int = 0) -> Any:
    from datasets import Dataset

    indices = list(range(start, start + n_rows))
    return Dataset.from_dict(
        {"idx": indices, "prompt": [f"p{index}" for index in indices]}
    )


def collate(rows: list[dict]) -> dict[str, torch.Tensor]:
    return {"idx": torch.tensor([int(row["idx"]) for row in rows], dtype=torch.long)}


def expected_columns(n_rows: int, start: int = 0) -> dict[str, list[float]]:
    """The values ``CountingRef`` produces for the whole dataset, in order."""
    indices = range(start, start + n_rows)
    return {
        "ref_chosen_logps": [float(index) for index in indices],
        "ref_rejected_logps": [-2.0 * index for index in indices],
    }


class CountingRef:
    """Reference callable recording which examples this rank actually scored."""

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        self.dtype = dtype
        self.examples_seen: list[int] = []

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        idx = batch["idx"]
        self.examples_seen.extend(int(value) for value in idx)
        values = idx.to(self.dtype)
        return {"ref_chosen_logps": values, "ref_rejected_logps": -2.0 * values}


_COLUMNS = ("ref_chosen_logps", "ref_rejected_logps")


def _assert_no_desync(world_size: int) -> None:
    """A follow-up collective must still succeed (proves no rank was left behind)."""
    token = torch.tensor([float(dist.get_rank() + 1)])
    dist.all_reduce(token, op=dist.ReduceOp.SUM)
    assert abs(token.item() - sum(range(1, world_size + 1))) < 1e-5


# --------------------------------------------------------------------------
# Workers
# --------------------------------------------------------------------------


def _worker_shards_and_restores_order(
    rank: int, world_size: int, port: int, n_rows: int, cache_dir: str
) -> None:
    """Each rank scores only its own shard; the result is the full dataset."""
    _setup_gloo(rank, world_size, port)
    try:
        from opaque.alignment.dpo.reference import compute_ref_logprobs_for_dataset
        from opaque.distributed import local_shard

        dataset = make_dataset(n_rows)
        ref = CountingRef()
        result = compute_ref_logprobs_for_dataset(
            dataset,
            ref,
            collate,
            _COLUMNS,
            cache_identity={"kind": "shard-order"},
            batch_size=2,
            cache_dir=cache_dir,
            use_cache=False,
        )

        own_shard = local_shard(range(n_rows), rank=rank, world_size=world_size)
        assert ref.examples_seen == list(own_shard), (
            f"rank {rank} scored {ref.examples_seen}, expected {list(own_shard)}"
        )

        expected = expected_columns(n_rows)
        assert len(result) == n_rows
        for name, values in expected.items():
            assert result[name] == values, f"rank {rank} column {name}: {result[name]}"

        _assert_no_desync(world_size)
    finally:
        _cleanup_ddp()


def _worker_empty_shard_preserves_dtype(
    rank: int, world_size: int, port: int, cache_dir: str
) -> None:
    """A rank with no rows must not force its dtype on the ranks that have some."""
    _setup_gloo(rank, world_size, port)
    try:
        from opaque.alignment.dpo.reference import compute_ref_logprobs_for_dataset

        # One row across two ranks: rank 0's shard is empty, rank 1 takes the
        # remainder and computes in bf16.  The row index is non-zero so the
        # assertion sees a real value rather than a dtype-independent zero.
        dataset = make_dataset(1, start=7)
        ref = CountingRef(dtype=torch.bfloat16)
        result = compute_ref_logprobs_for_dataset(
            dataset,
            ref,
            collate,
            _COLUMNS,
            cache_identity={"kind": "empty-shard"},
            batch_size=2,
            cache_dir=cache_dir,
            use_cache=False,
        )

        assert ref.examples_seen == ([] if rank == 0 else [7])
        assert result["ref_chosen_logps"] == [7.0]
        assert result["ref_rejected_logps"] == [-14.0]

        _assert_no_desync(world_size)
    finally:
        _cleanup_ddp()


def _worker_mismatched_dataset_sizes(rank: int, world_size: int, port: int) -> None:
    """Ranks holding different datasets are named as such, on every rank."""
    import pytest

    _setup_gloo(rank, world_size, port)
    try:
        from opaque.alignment.dpo.reference import compute_ref_logprobs_for_dataset

        with pytest.raises(RuntimeError, match="dataset size mismatch across ranks"):
            compute_ref_logprobs_for_dataset(
                make_dataset(4 if rank == 0 else 3),
                CountingRef(),
                collate,
                _COLUMNS,
                cache_identity={"kind": "size-mismatch"},
                batch_size=2,
                use_cache=False,
            )

        _assert_no_desync(world_size)
    finally:
        _cleanup_ddp()


def _worker_divergent_cache_state(
    rank: int, world_size: int, port: int, n_rows: int, cache_root: str
) -> None:
    """Node-local cache dirs must not split the group into different branches.

    Every rank gets its own cache directory and the seeding call writes on the
    main process only, so rank 0 alone holds the archive — the multi-node shape.
    """
    _setup_gloo(rank, world_size, port)
    try:
        from opaque.alignment.dpo.reference import compute_ref_logprobs_for_dataset

        dataset = make_dataset(n_rows)
        cache_dir = str(Path(cache_root) / f"rank{rank}")
        call = {
            "cache_identity": {"kind": "divergent-cache"},
            "batch_size": 2,
            "cache_dir": cache_dir,
            "use_cache": True,
        }

        compute_ref_logprobs_for_dataset(
            dataset, CountingRef(), collate, _COLUMNS, **call
        )
        archive_exists = any(Path(cache_dir).glob("*.safetensors"))
        assert archive_exists == (rank == 0), (
            f"rank {rank} unexpectedly {'has' if archive_exists else 'lacks'} an archive"
        )

        ref = CountingRef()
        result = compute_ref_logprobs_for_dataset(
            dataset, ref, collate, _COLUMNS, **call
        )

        # The group agreed to recompute, so every rank scored its own shard.
        assert ref.examples_seen, f"rank {rank} took the cache path alone"
        expected = expected_columns(n_rows)
        for name, values in expected.items():
            assert result[name] == values

        _assert_no_desync(world_size)
    finally:
        _cleanup_ddp()


def _worker_divergent_use_cache(
    rank: int, world_size: int, port: int, n_rows: int, cache_dir: str
) -> None:
    """Ranks disagreeing on ``use_cache`` must not deadlock.

    The cache decision is reduced whether or not the caller wants a cache, so
    the collective sequence does not depend on the argument.
    """
    _setup_gloo(rank, world_size, port)
    try:
        from opaque.alignment.dpo.reference import compute_ref_logprobs_for_dataset

        dataset = make_dataset(n_rows)
        ref = CountingRef()
        result = compute_ref_logprobs_for_dataset(
            dataset,
            ref,
            collate,
            _COLUMNS,
            cache_identity={"kind": "divergent-use-cache"},
            batch_size=2,
            cache_dir=cache_dir,
            use_cache=(rank == 0),
        )

        expected = expected_columns(n_rows)
        for name, values in expected.items():
            assert result[name] == values, f"rank {rank} column {name}: {result[name]}"

        _assert_no_desync(world_size)
    finally:
        _cleanup_ddp()


def _worker_shared_cache_hit(
    rank: int, world_size: int, port: int, n_rows: int, cache_dir: str
) -> None:
    """A cache every rank can see is reused without any rank calling ``ref``."""
    _setup_gloo(rank, world_size, port)
    try:
        from opaque.alignment.dpo.reference import compute_ref_logprobs_for_dataset

        dataset = make_dataset(n_rows)
        call = {
            "cache_identity": {"kind": "shared-cache"},
            "batch_size": 2,
            "cache_dir": cache_dir,
            "use_cache": True,
        }

        compute_ref_logprobs_for_dataset(
            dataset, CountingRef(), collate, _COLUMNS, **call
        )

        ref = CountingRef()
        result = compute_ref_logprobs_for_dataset(
            dataset, ref, collate, _COLUMNS, **call
        )

        assert ref.examples_seen == [], f"rank {rank} recomputed on a cache hit"
        expected = expected_columns(n_rows)
        for name, values in expected.items():
            assert result[name] == values

        _assert_no_desync(world_size)
    finally:
        _cleanup_ddp()
