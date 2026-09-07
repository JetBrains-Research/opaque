# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Fixed-population contract for ``DPTrainer`` under DDP.

``opaque.distributed.local_shard`` gives a remainder to the last rank when the
population is uneven. Stateful participation needs equal shard lengths, but
silently trimming a private tail is not stable under add/remove adjacency.
The trainer therefore accepts only populations divisible by ``world_size``.
"""

from __future__ import annotations

import dataclasses
import types

import pytest
import torch
from torch.utils.data import Dataset

from opaque.api.transformers.trainer._participation import (
    ResolvedParticipationPlan,
)
from opaque.exceptions import CheckpointError
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


def _shard_for(*, dataset_size: int, world_size: int, rank: int) -> Dataset:
    """Drive ``get_train_dataloader``'s shard path and return its dataset.

    Builds a real ``DPTrainer`` then pins ``_ddp`` to ``(rank, world_size)``
    and stubs ``_ctx`` so the dataloader factory takes the training branch
    (the inspection branch at ``ctx is None`` skips sharding).  The returned
    object is the rank-local dataset that the ``PoissonSampler`` would draw
    from.
    """
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
    )
    trainer = DPTrainer(model=model, args=args, train_dataset=dataset)
    trainer._ddp = dataclasses.replace(
        trainer._ddp,
        is_distributed=True,
        rank=rank,
        world_size=world_size,
    )
    effective_size = trainer._effective_train_dataset_size()
    expected_batch_size = args.per_device_train_batch_size * world_size
    participation = ResolvedParticipationPlan.resolve(
        mechanism_kind="gaussian",
        sampling_mode="poisson",
        sampling_kwargs={},
        population_size=effective_size,
        expected_batch_size=expected_batch_size,
        sample_rate=expected_batch_size / effective_size,
        total_steps=1,
        num_bins=1,
        world_size=world_size,
    )
    # Stub enough of ``_TrainingContext`` for the dataloader factory to
    # take the training-branch path that shards the dataset.
    trainer._ctx = types.SimpleNamespace(
        participation=participation,
        sample_rate=participation.sample_rate,
        expected_steps_per_epoch=1,
        total_steps=1,
        current_sampler=None,
        sampler_restart_step=None,
        mf=None,
        noise_multiplier=None,
        dataloader_in_order=True,
    )
    trainer._train_dataloader = None
    loader = trainer.get_train_dataloader()
    return loader.dataset


class TestEqualShardPopulation:
    @pytest.mark.parametrize(("dataset_size", "world_size"), [(10, 3), (5, 2)])
    def test_uneven_dataset_rejected(self, dataset_size, world_size):
        with pytest.raises(ValueError, match="divisible by world_size"):
            _shard_for(dataset_size=dataset_size, world_size=world_size, rank=0)

    def test_even_dataset_unchanged(self):
        """N=12, W=3 gives every rank exactly four records."""
        sizes = {
            rank: len(_shard_for(dataset_size=12, world_size=3, rank=rank))
            for rank in range(3)
        }
        assert sizes == {0: 4, 1: 4, 2: 4}

    def test_world_size_one_unchanged(self):
        """W=1 leaves the full population intact."""
        ds = _shard_for(dataset_size=10, world_size=1, rank=0)
        assert len(ds) == 10

    def test_smaller_than_world_size_rejected(self):
        """Every distributed rank must receive at least one record."""
        with pytest.raises(ValueError, match="fewer than world_size"):
            _shard_for(dataset_size=2, world_size=3, rank=0)


def _trainer_with_ddp(*, dataset_size: int, world_size: int) -> DPTrainer:
    """Build a DPTrainer pinned to ``world_size`` (rank 0)."""
    model = torch.nn.Linear(2, 2)
    dataset = _IdentityDataset(dataset_size)
    args = TrainingArguments(
        output_dir="/tmp/sample-rate-invariant-test",
        per_device_train_batch_size=2,
        clipping_norm=1.0,
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
        rank=0,
        world_size=world_size,
    )
    trainer.args.__dict__["world_size"] = world_size
    return trainer


class TestSampleRateInvariant:
    """``ctx.sample_rate`` must equal what the PoissonSampler ends up using.

    The accountant and sampler must use the same public, DDP-divisible global
    population. These tests pin that contract end-to-end.
    """

    def _setup(self, trainer: DPTrainer) -> None:
        # ``_setup_training`` is the single source of truth for
        # ``ctx.sample_rate``; we drive it directly to keep the test fast
        # (no full ``train()`` loop required).  ``train()`` would otherwise
        # store the returned ctx on ``self._ctx``, so we mirror that.
        trainer._ctx = trainer._setup_training()

    def test_sample_rate_uses_validated_global_denominator(self):
        # N=12, W=3 and global expected batch 2/rank * 3 = 6 gives q=1/2.
        trainer = _trainer_with_ddp(dataset_size=12, world_size=3)
        self._setup(trainer)
        assert trainer._ctx.sample_rate == pytest.approx(6 / 12)

    def test_sampler_q_matches_accountant_q(self):
        # End-to-end: ``ctx.sample_rate`` (accountant view) must equal the
        # rate the constructed sampler is configured with (sampler view).
        # ``ctx.current_sampler`` is the single ``PoissonSampler``
        # instance for the whole training run.
        trainer = _trainer_with_ddp(dataset_size=12, world_size=3)
        self._setup(trainer)
        trainer.get_train_dataloader()
        sampler_rate = trainer._ctx.current_sampler.sample_rate
        assert sampler_rate == pytest.approx(trainer._ctx.sample_rate)

    def test_world_size_one_uses_full_denominator(self):
        # W=1 uses q = 2/10. Both views agree, no drift.
        trainer = _trainer_with_ddp(dataset_size=10, world_size=1)
        self._setup(trainer)
        assert trainer._ctx.sample_rate == pytest.approx(2 / 10)
        trainer.get_train_dataloader()
        assert trainer._ctx.current_sampler.sample_rate == pytest.approx(
            trainer._ctx.sample_rate
        )

    def test_uneven_population_rejected_before_accounting_or_sampling(self):
        trainer = _trainer_with_ddp(dataset_size=10, world_size=3)

        with pytest.raises(ValueError, match="divisible by world_size"):
            self._setup(trainer)

    def test_truncated_poisson_is_rejected_before_ddp_accounting_or_sampling(self):
        trainer = _trainer_with_ddp(dataset_size=12, world_size=3)
        trainer.args.sampling_kwargs = {"truncated_batch_size": 2}

        with pytest.raises(
            ValueError,
            match="truncated Poisson sampling is not supported with distributed",
        ):
            self._setup(trainer)

    def test_ddp_sampler_restore_fails_closed_without_per_rank_snapshots(
        self, tmp_path
    ):
        trainer = _trainer_with_ddp(dataset_size=12, world_size=3)

        with pytest.raises(
            CheckpointError,
            match="Distributed checkpoint resume cannot restore the shared rank-0",
        ):
            trainer._train_once(
                resume_from_checkpoint=str(tmp_path),
                microbatch_size_override=None,
                ignore_keys_for_eval=None,
            )
