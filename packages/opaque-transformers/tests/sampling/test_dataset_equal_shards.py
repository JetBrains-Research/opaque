# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Pre-trim contract for ``DPTrainer.get_train_dataloader`` under DDP.

``opaque.distributed.local_shard`` hands the remainder examples to the
last rank when ``len(dataset) % world_size != 0``.  That's harmless for
Poisson sampling but desynchronises the batch count across ranks for
fixed-order samplers (BLT-sequential, BnB) that FTRL integrations will
hand to ``get_train_dataloader`` later — the trainer must therefore trim
the dataset to a multiple of ``world_size`` before sharding so every
rank ends up with an identically sized shard.
"""

from __future__ import annotations

import dataclasses
import types

import pytest
import torch
from torch.utils.data import Dataset

from opaque.transformers.trainer import DPTrainer, DPTrainingArguments


pytest.importorskip("transformers")


class _IdentityDataset(Dataset):
    """Tiny dataset that returns its integer index as ``input_ids`` / ``labels``."""

    def __init__(self, n: int) -> None:
        self._n = int(n)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        v = int(idx)
        return {
            "input_ids": torch.tensor([v, v], dtype=torch.long),
            "labels": torch.tensor([v, v], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1], dtype=torch.long),
        }


def _shard_for(
    *, dataset_size: int, world_size: int, rank: int
) -> Dataset:
    """Drive ``get_train_dataloader``'s shard path and return its dataset.

    Builds a real ``DPTrainer`` then pins ``_ddp`` to ``(rank, world_size)``
    and stubs ``_ctx`` so the dataloader factory takes the training branch
    (the inspection branch at ``ctx is None`` skips sharding).  The returned
    object is the rank-local dataset that the ``PoissonSampler`` would draw
    from.
    """
    model = torch.nn.Linear(2, 2)
    dataset = _IdentityDataset(dataset_size)
    args = DPTrainingArguments(
        output_dir="/tmp/equal-shards-test",
        per_device_train_batch_size=2,
        max_grad_norm=1.0,
        privacy_target_epsilon=10.0,
        privacy_noise_multiplier=1.0,
        max_steps=1,
        save_strategy="no",
        report_to=[],
        use_cpu=True,
    )
    trainer = DPTrainer(model=model, args=args, train_dataset=dataset)
    trainer._ddp = dataclasses.replace(
        trainer._ddp,
        is_distributed=True,
        rank=rank,
        world_size=world_size,
    )
    # Stub enough of ``_TrainingContext`` for the dataloader factory to
    # take the training-branch path that shards the dataset.
    trainer._ctx = types.SimpleNamespace(
        sample_rate=0.5,
        expected_steps_per_epoch=1,
        current_sampler=None,
    )
    trainer._train_dataloader = None
    loader = trainer.get_train_dataloader()
    return loader.dataset


class TestEqualShardTrim:
    def test_uneven_dataset_trims_last_rank(self):
        """N=10, W=3, rank=2 should see 3 examples (floor(10/3)), not 4."""
        ds = _shard_for(dataset_size=10, world_size=3, rank=2)
        assert len(ds) == 3

    def test_uneven_dataset_keeps_first_rank_equal(self):
        """N=10, W=3, rank=0 already sees floor(10/3)=3 examples — unchanged."""
        ds = _shard_for(dataset_size=10, world_size=3, rank=0)
        assert len(ds) == 3

    def test_uneven_dataset_makes_all_ranks_equal(self):
        """N=10, W=3 → every rank's shard has length 3."""
        sizes = {
            r: len(_shard_for(dataset_size=10, world_size=3, rank=r))
            for r in range(3)
        }
        assert sizes == {0: 3, 1: 3, 2: 3}

    def test_even_dataset_unchanged(self):
        """N=12, W=3 — already a multiple, no trim, last rank sees 4."""
        ds = _shard_for(dataset_size=12, world_size=3, rank=2)
        assert len(ds) == 4

    def test_world_size_one_unchanged(self):
        """W=1 — no sharding, no trim, dataloader sees the whole dataset."""
        ds = _shard_for(dataset_size=10, world_size=1, rank=0)
        assert len(ds) == 10
