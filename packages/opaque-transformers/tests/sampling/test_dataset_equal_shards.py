# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Equal-shard population contract for ``DPTrainer`` under DDP.

``opaque.distributed.local_shard`` hands the remainder examples to the
last rank when ``len(dataset) % world_size != 0``.  That's harmless for
Poisson sampling but desynchronises the batch count across ranks for
fixed-order samplers (BLT-sequential, balls-in-bins) that FTRL
integrations hand to ``get_train_dataloader``.

An earlier revision resolved this by silently trimming
``len(train_dataset) % world_size`` tail examples before sharding. Under
add/remove adjacency that trim is data-dependent: neighboring datasets of
length ``N`` and ``N - 1`` can floor-divide to *different* multiples of
``world_size``, so both the excluded tail and the privacy-accounting
sample-rate denominator become a function of the private dataset length —
exactly the kind of data-dependent mechanism selection differential
privacy must avoid (see ``.junie/differential-privacy-review.md``, "Query
and sensitivity").

``DPTrainer`` now fails closed instead:
``DPTrainer._effective_train_dataset_size()`` requires
``len(train_dataset)`` to already be an exact multiple of ``world_size``
(a value fixed by run configuration, not by any single record) and raises
``ConfigurationError`` otherwise. These tests pin the divisible-success
path (equal shards, stable sample-rate denominator) and the
non-divisible-rejection path (fail-closed, no silent trim).
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
    *, dataset_size: int, world_size: int, rank: int = 0
) -> DPTrainer:
    """Build a real ``DPTrainer`` then pin ``_ddp`` to ``(rank, world_size)``.

    Construction itself uses the default (non-distributed) ``_ddp``
    resolved from the environment, so it never trips the population
    check; the check only fires once a test explicitly overrides
    ``_ddp.world_size``, mirroring how a real multi-rank launch resolves
    ``world_size`` after the trainer object exists.
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
        is_distributed=world_size > 1,
        rank=rank,
        world_size=world_size,
    )
    return trainer


def _shard_for(*, dataset_size: int, world_size: int, rank: int) -> Dataset:
    """Drive ``get_train_dataloader``'s shard path and return its dataset.

    Stubs ``_ctx`` so the dataloader factory takes the training branch
    (the inspection branch at ``ctx is None`` skips sharding). The
    returned object is the rank-local dataset the ``PoissonSampler``
    would draw from.
    """
    trainer = _trainer_with_ddp(
        dataset_size=dataset_size, world_size=world_size, rank=rank
    )
    # Stub enough of ``_TrainingContext`` for the dataloader factory to
    # take the training-branch path that shards the dataset.
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
    loader = trainer.get_train_dataloader()
    return loader.dataset


class TestEqualShardValidation:
    """Divisible populations shard evenly; non-divisible ones fail closed."""

    def test_evenly_divisible_dataset_shards_equally(self):
        """N=12, W=3 — already a multiple, every rank sees 4 examples."""
        sizes = {
            r: len(_shard_for(dataset_size=12, world_size=3, rank=r)) for r in range(3)
        }
        assert sizes == {0: 4, 1: 4, 2: 4}

    def test_world_size_one_unchanged(self):
        """W=1 — no sharding, no validation, dataloader sees the whole dataset."""
        ds = _shard_for(dataset_size=10, world_size=1, rank=0)
        assert len(ds) == 10

    def test_non_divisible_dataset_rejected(self):
        """N=10, W=3 (10 % 3 == 1) must raise instead of silently trimming to 9."""
        with pytest.raises(ConfigurationError, match="not evenly divisible"):
            _shard_for(dataset_size=10, world_size=3, rank=0)

    def test_dataset_smaller_than_world_size_rejected(self):
        """N=2, W=3 is non-divisible (2 % 3 == 2) — rejected, not silently emptied."""
        with pytest.raises(ConfigurationError, match="not evenly divisible"):
            _shard_for(dataset_size=2, world_size=3, rank=0)

    def test_empty_dataset_under_ddp_rejected(self):
        """N=0, W=3 divides evenly (0 % 3 == 0) but leaves every rank with 0 examples."""
        with pytest.raises(ConfigurationError, match="empty"):
            _shard_for(dataset_size=0, world_size=3, rank=0)

    def test_rejection_message_suggests_compatible_sizes(self):
        """The error names concrete divisible sizes so the fix is actionable."""
        trainer = _trainer_with_ddp(dataset_size=10, world_size=3)
        with pytest.raises(ConfigurationError, match=r"\(e\.g\. 9 or 12 examples\)"):
            trainer._effective_train_dataset_size()


class TestSampleRateInvariant:
    """``ctx.sample_rate`` must equal what the PoissonSampler ends up using.

    Because the population is now validated rather than trimmed, the
    accounting denominator is exactly ``len(train_dataset)`` under DDP
    (never a data-dependent floor-divided value) — these tests pin that
    contract end-to-end: drive ``_setup_training`` so the real
    ``ctx.sample_rate`` is computed, then drive ``get_train_dataloader``
    and compare the sampler's stored rate to it.
    """

    def _setup(self, trainer: DPTrainer) -> None:
        # ``_setup_training`` is the single source of truth for
        # ``ctx.sample_rate``; we drive it directly to keep the test fast
        # (no full ``train()`` loop required).  ``train()`` would otherwise
        # store the returned ctx on ``self._ctx``, so we mirror that.
        trainer._ctx = trainer._setup_training()

    def test_sample_rate_uses_full_denominator(self):
        # N=12, W=3 is already divisible; expected_batch_size=2 → q = 2/12,
        # not a data-dependent floor-divided value.
        trainer = _trainer_with_ddp(dataset_size=12, world_size=3)
        self._setup(trainer)
        assert trainer._ctx.sample_rate == pytest.approx(2 / 12)

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
        # W=1 → no DDP validation → q = 2/10.  Both views agree, no drift.
        trainer = _trainer_with_ddp(dataset_size=10, world_size=1)
        self._setup(trainer)
        assert trainer._ctx.sample_rate == pytest.approx(2 / 10)
        trainer.get_train_dataloader()
        assert trainer._ctx.current_sampler.sample_rate == pytest.approx(
            trainer._ctx.sample_rate
        )

    def test_non_divisible_dataset_rejected_before_setup(self):
        # N=10, W=3 must fail closed at ``_setup_training`` time — the
        # accountant must never calibrate against a trimmed, data-dependent
        # denominator.
        trainer = _trainer_with_ddp(dataset_size=10, world_size=3)
        with pytest.raises(ConfigurationError, match="not evenly divisible"):
            self._setup(trainer)
