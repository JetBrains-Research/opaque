# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""``DPTrainer`` population validation under DDP.

Divisible dataset sizes shard evenly; non-divisible sizes raise
``ConfigurationError`` unless ``ddp_drop_uneven_population=True`` opts into
trimming instead.
"""

from __future__ import annotations

import dataclasses
import types

import pytest
import torch
from torch.utils.data import Dataset

from opaque.exceptions import ConfigurationError
from opaque.transformers.trainer import DPTrainer, TrainingArguments

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


def _trainer_with_ddp(
    *,
    dataset_size: int,
    world_size: int,
    rank: int = 0,
    ddp_drop_uneven_population: bool = False,
) -> DPTrainer:
    """Build a ``DPTrainer`` and pin ``_ddp`` to ``(rank, world_size)``."""
    model = torch.nn.Linear(2, 2)
    dataset = _IdentityDataset(dataset_size)
    args = TrainingArguments(
        output_dir="/tmp/equal-shards-test",
        per_device_train_batch_size=2,
        clipping_norm=1.0,
        privacy_target_epsilon=10.0,
        privacy_noise_multiplier=1.0,
        max_steps=1,
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        ddp_drop_uneven_population=ddp_drop_uneven_population,
    )
    trainer = DPTrainer(model=model, args=args, train_dataset=dataset)
    trainer._ddp = dataclasses.replace(
        trainer._ddp,
        is_distributed=world_size > 1,
        rank=rank,
        world_size=world_size,
    )
    return trainer


def _shard_for(
    *,
    dataset_size: int,
    world_size: int,
    rank: int,
    ddp_drop_uneven_population: bool = False,
) -> Dataset:
    """Drive ``get_train_dataloader``'s sharding branch; return its dataset."""
    trainer = _trainer_with_ddp(
        dataset_size=dataset_size,
        world_size=world_size,
        rank=rank,
        ddp_drop_uneven_population=ddp_drop_uneven_population,
    )
    trainer._ctx = types.SimpleNamespace(
        sample_rate=0.5,
        expected_steps_per_epoch=1,
        total_steps=1,
        current_sampler=None,
        sampler_restart_step=None,
        mf=None,
        noise_multiplier=None,
    )
    trainer._train_dataloader = None
    return trainer.get_train_dataloader().dataset


def test_divisible_dataset_shards_equally():
    """N=12, W=3 — already a multiple, every rank gets 4 examples."""
    sizes = {
        r: len(_shard_for(dataset_size=12, world_size=3, rank=r)) for r in range(3)
    }
    assert sizes == {0: 4, 1: 4, 2: 4}


def test_non_divisible_dataset_rejected():
    """N=10, W=3 (10 % 3 == 1) raises instead of silently trimming to 9."""
    with pytest.raises(ConfigurationError, match="not evenly divisible"):
        _shard_for(dataset_size=10, world_size=3, rank=0)


def test_non_divisible_dataset_trimmed_when_opted_in():
    """N=10, W=3 with ddp_drop_uneven_population=True trims to 9 (3/rank)."""
    sizes = {
        r: len(
            _shard_for(
                dataset_size=10,
                world_size=3,
                rank=r,
                ddp_drop_uneven_population=True,
            )
        )
        for r in range(3)
    }
    assert sizes == {0: 3, 1: 3, 2: 3}


def test_sampler_rate_matches_accountant_rate():
    """The sampler's configured rate equals the accountant's ``sample_rate``."""
    trainer = _trainer_with_ddp(dataset_size=12, world_size=3)
    trainer._ctx = trainer._setup_training()
    trainer.get_train_dataloader()
    assert trainer._ctx.current_sampler.sample_rate == pytest.approx(
        trainer._ctx.sample_rate
    )
