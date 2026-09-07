"""Subprocess-launchable DDP test runner for DPTrainer.

Launched by ``test_ddp_trainer.py`` via ``subprocess.Popen`` (one process
per rank). Receives ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` /
``MASTER_ADDR`` / ``MASTER_PORT`` from the env, runs a self-contained
DDP scenario, and exits with a non-zero code on any assertion failure
(stderr captures details).

This avoids ``mp.spawn`` because pytest's ``--import-mode=importlib`` mode
renames test modules in a way the spawned worker can't unpickle.

Self-contained tiny PreTrainedModel subclass so we sidestep the HF
attention path's ``vmap`` incompatibility with transformers 5.x — the
bug is in :mod:`transformers.masking_utils._ignore_causal_mask_sdpa`,
unrelated to the DDP path under test.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import PretrainedConfig, PreTrainedModel, TrainerCallback
from transformers.modeling_outputs import CausalLMOutput

from opaque.exceptions import OperationError
from opaque.transformers.trainer import DPTrainer, TrainingArguments


class TinyConfig(PretrainedConfig):
    # This test-only family implements a vmap-compatible forward directly.
    # It is intentionally not part of opaque's production patch registry.
    model_type = "tiny_dp"

    def __init__(self, vocab_size: int = 64, hidden_size: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size


class TinyForCausalLM(PreTrainedModel):
    """Tiny next-token model: embed -> linear -> cross-entropy.

    Shape: input_ids (B, L) -> logits (B, L, V).  Avoids HF's masking /
    SDPA / RoPE plumbing entirely so per-example vmap'd gradients work
    without hitting the transformers-5.x `_ignore_causal_mask_sdpa`
    bug.
    """

    config_class = TinyConfig
    main_input_name = "input_ids"

    def __init__(self, config: TinyConfig):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, labels=None, **_unused):
        h = self.embed(input_ids)
        logits = self.head(h)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(loss=loss, logits=logits)


class TinyDataset(Dataset):
    """Deterministic synthetic dataset of length-`seq_len` token sequences."""

    def __init__(self, n: int, seq_len: int, vocab: int, seed: int = 0):
        gen = torch.Generator().manual_seed(seed)
        self._data = [
            torch.randint(0, vocab, (seq_len,), generator=gen) for _ in range(n)
        ]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        ids = self._data[idx]
        return {"input_ids": ids, "labels": ids.clone()}


def _collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    # Poisson rounds occasionally produce empty batches at low sample
    # rates / per-rank shards.  Return zero-row tensors so downstream
    # code (clipped_grad, DDP collectives) takes its empty-batch path:
    # ``clipped_grad`` short-circuits internally, ``sum_gradients_``
    # all-reduces zero gradients, and ``training_step`` reports
    # ``batch_size=0`` based on the synced ``aux.batch_size``.
    if len(batch) == 0:
        return {
            "input_ids": torch.zeros((0, 1), dtype=torch.long),
            "labels": torch.zeros((0, 1), dtype=torch.long),
        }
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def _setup_ddp(
    rank: int, world_size: int, port: int, backend: str | None = None
) -> torch.device:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    if backend is None and torch.cuda.is_available():
        torch.cuda.set_device(rank)
        backend = "nccl"
        device = torch.device(f"cuda:{rank}")
    elif backend is None:
        backend = "gloo"
        device = torch.device("cpu")
    elif backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return device


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def scenario_runtime_foundation(
    rank: int,
    world_size: int,
    output_dir: str,
    use_cpu: bool = False,
    **_,
) -> None:
    """Verify rank/world plumbing + checkpoint gating."""
    cfg = TinyConfig()
    model = TinyForCausalLM(cfg)
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        max_steps=2,
        logging_steps=1,
        save_steps=1,
        save_strategy="steps",
        save_total_limit=2,
        seed=7,
        privacy_target_epsilon=8.0,
        privacy_target_delta=1e-5,
        report_to=[],
        use_cpu=use_cpu,
        use_compat_patches=False,
    )
    ds = TinyDataset(n=32, seq_len=8, vocab=cfg.vocab_size)
    trainer = DPTrainer(
        model=model, args=args, train_dataset=ds, data_collator=_collate
    )

    # Rank/world plumbing.
    assert trainer._ddp.is_distributed
    assert trainer._ddp.rank == rank
    assert trainer._ddp.world_size == world_size
    assert trainer.is_world_process_zero() == (rank == 0)

    # State carries the rank flags so HF callbacks can gate themselves.
    assert trainer.state.is_world_process_zero == (rank == 0)
    assert trainer.state.is_local_process_zero == (rank == 0)

    # Run training.  Each rank computes the same cluster-wide loss because
    # aux is sync'd and the gradient is AllReduce'd.
    out = trainer.train()
    assert out is not None

    # Checkpoint gating: after training, output_dir should contain
    # exactly one checkpoint-{step} dir written by rank 0, and per-rank
    # rng_state_{rank}.pth files written by every rank.
    dist.barrier()
    if rank == 0:
        children = sorted(p.name for p in Path(output_dir).iterdir())
        ckpts = [c for c in children if c.startswith("checkpoint-")]
        assert ckpts, f"No checkpoints in {output_dir}: {children}"
        ckpt_dir = Path(output_dir) / ckpts[-1]
        files = {p.name for p in ckpt_dir.iterdir()}
        # Rank-0-only artefacts.
        assert "trainer_state.json" in files
        # Per-rank RNG snapshots written by every rank.
        for r in range(world_size):
            assert f"rng_state_{r}.pth" in files, (
                f"missing rank {r} rng snapshot in {files}"
            )


def scenario_per_rank_partition(rank: int, world_size: int, **_) -> None:
    """Verify ``local_shard`` partitions the dataset across ranks.

    Build a sampler with seed S; collect all indices yielded across one
    epoch; gather to rank 0; assert disjoint and union = range(N).
    """
    from opaque.distributed import local_shard

    full_n = 64
    seq = 8
    cfg = TinyConfig()
    full_ds = TinyDataset(n=full_n, seq_len=seq, vocab=cfg.vocab_size)
    shard = local_shard(full_ds, rank=rank, world_size=world_size)

    # Indices the sampler yields are *local* to the shard; recover
    # global indices via the Subset's `.indices` view.
    global_indices = list(shard.indices)
    # Verify shards are contiguous and disjoint.
    gathered = [None] * world_size
    dist.all_gather_object(gathered, global_indices)
    if rank == 0:
        flat = sorted(i for shard_indices in gathered for i in shard_indices)
        assert flat == list(range(full_n)), (
            f"shards do not partition range({full_n}): {flat[:10]}…"
        )


def _run_eval_gather_case(
    rank: int,
    output_dir: str,
    *,
    eval_size: int,
    use_cpu: bool,
) -> None:
    """Compare distributed evaluation with a full-dataset model reference."""
    torch.manual_seed(1234)
    cfg = TinyConfig(vocab_size=32, hidden_size=8)
    model = TinyForCausalLM(cfg)
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        max_steps=1,
        save_strategy="no",
        report_to=[],
        seed=11,
        privacy_noise_multiplier=0.0,
        use_cpu=use_cpu,
        use_compat_patches=False,
    )
    train_ds = TinyDataset(n=16, seq_len=4, vocab=cfg.vocab_size)
    eval_ds = TinyDataset(n=eval_size, seq_len=4, vocab=cfg.vocab_size, seed=99)
    captured = {}

    def compute_metrics(ep):
        predictions = torch.from_numpy(ep.predictions)
        labels = torch.from_numpy(ep.label_ids)
        captured["predictions"] = predictions
        captured["labels"] = labels
        return {"accuracy": float((predictions.argmax(-1) == labels).float().mean())}

    trainer = DPTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=_collate,
        compute_metrics=compute_metrics,
    )

    device = next(model.parameters()).device
    full_batch = {
        key: value.to(device)
        for key, value in _collate([eval_ds[i] for i in range(len(eval_ds))]).items()
    }
    with torch.no_grad():
        reference = model(**full_batch)
    reference_predictions = reference.logits.detach().cpu()
    reference_labels = full_batch["labels"].cpu()
    reference_accuracy = float(
        (reference_predictions.argmax(-1) == reference_labels).float().mean()
    )
    reference_loss = reference.loss.item()

    metrics = trainer.evaluate()
    assert captured["predictions"].shape[0] == eval_size
    assert torch.equal(captured["labels"], reference_labels)
    assert torch.allclose(
        captured["predictions"],
        reference_predictions,
        atol=2e-7,
        rtol=1e-5,
    )
    assert abs(metrics["eval_accuracy"] - reference_accuracy) < 1e-7
    assert abs(metrics["eval_loss"] - reference_loss) < 1e-6


def scenario_eval_gather(
    rank: int,
    output_dir: str,
    use_cpu: bool = False,
    **_,
) -> None:
    """Verify evaluation gathers uneven rank-local shards in dataset order."""
    _run_eval_gather_case(rank, output_dir, eval_size=5, use_cpu=use_cpu)


def scenario_eval_gather_empty_rank(
    rank: int,
    output_dir: str,
    use_cpu: bool = False,
    **_,
) -> None:
    """Verify evaluation gathers when one rank receives no examples."""
    _run_eval_gather_case(rank, output_dir, eval_size=1, use_cpu=use_cpu)


def scenario_batch_eval_metrics(
    rank: int, world_size: int, output_dir: str, use_cpu: bool = False, **_
) -> None:
    """Verify DDP ``batch_eval_metrics`` runs on gathered batch payloads."""
    cfg = TinyConfig(vocab_size=32, hidden_size=8)
    model = TinyForCausalLM(cfg)
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        max_steps=1,
        save_strategy="no",
        report_to=[],
        seed=17,
        batch_eval_metrics=True,
        include_for_metrics=["inputs", "loss"],
        privacy_noise_multiplier=0.0,
        use_cpu=use_cpu,
        use_compat_patches=False,
    )
    train_ds = TinyDataset(n=16, seq_len=4, vocab=cfg.vocab_size)
    eval_ds = TinyDataset(n=20, seq_len=4, vocab=cfg.vocab_size, seed=101)
    running = {"seen": 0}

    def compute_metrics(ep, compute_result: bool = False):
        assert ep.predictions is not None
        assert ep.label_ids is not None
        if ep.inputs is not None:
            assert isinstance(ep.inputs, dict)
            assert ep.inputs["input_ids"].shape[0] == ep.predictions.shape[0]
        if ep.losses is not None:
            assert ep.losses.shape[0] == ep.predictions.shape[0]
        running["seen"] += int(ep.predictions.shape[0])
        if compute_result:
            return {"seen": float(running["seen"])}
        return {}

    trainer = DPTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=_collate,
        compute_metrics=compute_metrics,
    )
    metrics = trainer.evaluate()
    if rank == 0:
        assert int(metrics["eval_seen"]) == len(eval_ds), metrics


def scenario_rank_gating_and_worker_seed(
    rank: int, world_size: int, output_dir: str, use_cpu: bool = False, **_
) -> None:
    """Verify rank-gated logging/saving and worker seed rank wiring."""
    cfg = TinyConfig()
    model = TinyForCausalLM(cfg)
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        max_steps=1,
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=2,
        privacy_noise_multiplier=0.0,
        use_cpu=use_cpu,
        use_compat_patches=False,
    )
    trainer = DPTrainer(
        model=model,
        args=args,
        train_dataset=TinyDataset(8, 4, cfg.vocab_size),
        data_collator=_collate,
    )

    worker_init = trainer._dataloader_worker_init_fn()
    assert worker_init is not None
    assert worker_init.keywords is not None
    assert worker_init.keywords["rank"] == rank

    trainer.log({"smoke": float(rank)})
    trainer.save_metrics("rank_gate", {"rank": float(rank)}, combined=False)
    trainer.save_state()

    dist.barrier()
    metrics_path = Path(output_dir) / "rank_gate_results.json"
    state_path = Path(output_dir) / "trainer_state.json"
    with metrics_path.open() as f:
        saved_metrics = json.load(f)
    if rank == 0:
        assert len(trainer.state.log_history) == 1
        assert metrics_path.exists()
        assert state_path.exists()
        assert saved_metrics["rank"] == 0.0
    else:
        assert len(trainer.state.log_history) == 0
        assert metrics_path.exists()
        assert state_path.exists()
        assert saved_metrics["rank"] == 0.0


def scenario_gather_paths(rank: int, world_size: int, **_) -> None:
    """Verify gather fastpath and object fallback both return global payloads."""
    from opaque.api.engine.distributed._state import gather_pytree, gather_tensors

    same = torch.full((2, 3), float(rank), dtype=torch.float32)
    gathered_same = gather_tensors(same, dim=0)
    assert gathered_same.shape == (2 * world_size, 3)
    for r in range(world_size):
        chunk = gathered_same[r * 2 : (r + 1) * 2]
        assert torch.allclose(chunk, torch.full_like(chunk, float(r)))

    ragged = torch.full((rank + 1, 2), float(rank), dtype=torch.float32)
    gathered_ragged = gather_tensors(ragged, dim=0)
    expected_rows = sum(i + 1 for i in range(world_size))
    assert gathered_ragged.shape == (expected_rows, 2)
    row_offset = 0
    for r in range(world_size):
        rows = r + 1
        chunk = gathered_ragged[row_offset : row_offset + rows]
        assert torch.allclose(chunk, torch.full_like(chunk, float(r)))
        row_offset += rows

    pytree = {"pred": same, "aux": None}
    gathered_tree = gather_pytree(pytree)
    assert gathered_tree["aux"] is None
    assert gathered_tree["pred"].shape == (2 * world_size, 3)


def scenario_step_authority_consensus(
    rank: int,
    world_size: int,
    output_dir: str,
    use_cpu: bool = False,
    **_,
) -> None:
    """A rank-0-only on-save mutation fails uniformly instead of deadlocking."""
    cfg = TinyConfig()
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        max_steps=1,
        save_steps=1,
        save_strategy="steps",
        report_to=[],
        privacy_noise_multiplier=1.0,
        use_cpu=use_cpu,
        use_compat_patches=False,
    )

    class _ForgeProgressOnSavingRank(TrainerCallback):
        def on_save(self, args_, state_, control_, **_kwargs):
            assert rank == 0
            state_.global_step = 99

    trainer = DPTrainer(
        model=TinyForCausalLM(cfg),
        args=args,
        train_dataset=TinyDataset(16, 4, cfg.vocab_size),
        data_collator=_collate,
        callbacks=[_ForgeProgressOnSavingRank()],
    )
    failed = False
    try:
        trainer.train()
    except OperationError as exc:
        failed = "executed-step authority" in str(exc)
    failures = [None] * world_size
    dist.all_gather_object(failures, failed)
    assert failures == [True] * world_size, failures


def scenario_post_optimizer_oom_consensus(
    rank: int,
    world_size: int,
    output_dir: str,
    use_cpu: bool = False,
    **_,
) -> None:
    """A rank-local post-noise OOM aborts every rank without replay."""
    cfg = TinyConfig()
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        microbatch_size=2,
        auto_find_microbatch_size=True,
        max_steps=2,
        save_strategy="no",
        report_to=[],
        privacy_noise_multiplier=1.0,
        use_cpu=use_cpu,
        use_compat_patches=False,
    )
    callback_calls = 0

    class _RankZeroOptimizerOom(TrainerCallback):
        def on_optimizer_step(self, args_, state_, control_, **_kwargs):
            nonlocal callback_calls
            callback_calls += 1
            if rank == 0:
                raise torch.OutOfMemoryError("rank-zero post-optimizer OOM")

    trainer = DPTrainer(
        model=TinyForCausalLM(cfg),
        args=args,
        train_dataset=TinyDataset(16, 4, cfg.vocab_size),
        data_collator=_collate,
        callbacks=[_RankZeroOptimizerOom()],
    )
    failed = False
    try:
        trainer.train()
    except OperationError as exc:
        failed = "privacy release became observable" in str(exc)
    outcomes = [None] * world_size
    dist.all_gather_object(outcomes, (failed, callback_calls))
    assert outcomes == [(True, 1)] * world_size, outcomes


def scenario_input_oom_retry_consensus(
    rank: int,
    world_size: int,
    output_dir: str,
    use_cpu: bool = False,
    **_,
) -> None:
    """A rank-local input-transfer OOM retries at one aligned boundary."""
    cfg = TinyConfig()
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        microbatch_size=2,
        auto_find_microbatch_size=True,
        max_steps=1,
        save_strategy="no",
        report_to=[],
        privacy_noise_multiplier=1.0,
        use_cpu=use_cpu,
        use_compat_patches=False,
    )
    trainer = DPTrainer(
        model=TinyForCausalLM(cfg),
        args=args,
        train_dataset=TinyDataset(16, 4, cfg.vocab_size),
        data_collator=_collate,
    )
    original_prepare = trainer._prepare_input
    raised = False

    def _oom_once(inputs):
        nonlocal raised
        if rank == 0 and trainer._ctx is not None and not raised:
            raised = True
            raise torch.OutOfMemoryError("rank-zero input preparation OOM")
        return original_prepare(inputs)

    trainer._prepare_input = _oom_once
    result = trainer.train()
    outcomes = [None] * world_size
    dist.all_gather_object(
        outcomes,
        (result.global_step, trainer.state.converged_microbatch_size, raised),
    )
    assert outcomes == [(1, 1, True), (1, 1, False)], outcomes


def scenario_metric_oom_consensus(
    rank: int,
    world_size: int,
    output_dir: str,
    use_cpu: bool = False,
    **_,
) -> None:
    """A rank-local post-update metric OOM aborts without replay."""
    cfg = TinyConfig()
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        microbatch_size=2,
        auto_find_microbatch_size=True,
        max_steps=2,
        save_strategy="no",
        report_to=[],
        privacy_noise_multiplier=1.0,
        use_cpu=use_cpu,
        use_compat_patches=False,
    )
    trainer = DPTrainer(
        model=TinyForCausalLM(cfg),
        args=args,
        train_dataset=TinyDataset(16, 4, cfg.vocab_size),
        data_collator=_collate,
    )
    original_metrics = trainer._build_training_step_metrics
    calls = 0

    def _oom_on_rank_zero(grads, noisy_grads, aux):
        nonlocal calls
        calls += 1
        if rank == 0:
            raise torch.OutOfMemoryError("rank-zero metric construction OOM")
        return original_metrics(grads, noisy_grads, aux)

    trainer._build_training_step_metrics = _oom_on_rank_zero
    failed = False
    try:
        trainer.train()
    except OperationError as exc:
        failed = "privacy release became observable" in str(exc)
    outcomes = [None] * world_size
    dist.all_gather_object(outcomes, (failed, calls))
    assert outcomes == [(True, 1)] * world_size, outcomes


def scenario_token_reduce_prep_oom_consensus(
    rank: int,
    world_size: int,
    output_dir: str,
    use_cpu: bool = False,
    **_,
) -> None:
    """A rank-local OOM before token SUM aborts before any rank enters it."""
    cfg = TinyConfig()
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        microbatch_size=2,
        auto_find_microbatch_size=True,
        max_steps=1,
        save_strategy="no",
        report_to=[],
        privacy_noise_multiplier=1.0,
        include_num_input_tokens_seen="all",
        average_tokens_across_devices=True,
        use_cpu=use_cpu,
        use_compat_patches=False,
    )
    trainer = DPTrainer(
        model=TinyForCausalLM(cfg),
        args=args,
        train_dataset=TinyDataset(16, 4, cfg.vocab_size),
        data_collator=_collate,
    )
    original_prepare = trainer._prepare_token_count_reduction
    calls = 0

    def _oom_on_rank_zero(n_tokens):
        nonlocal calls
        calls += 1
        if rank == 0:
            raise torch.OutOfMemoryError("rank-zero token reduction preparation OOM")
        return original_prepare(n_tokens)

    trainer._prepare_token_count_reduction = _oom_on_rank_zero
    failed = False
    try:
        trainer.train()
    except OperationError as exc:
        failed = "privacy release became observable" in str(exc)
    outcomes = [None] * world_size
    dist.all_gather_object(outcomes, (failed, calls))
    assert outcomes == [(True, 1)] * world_size, outcomes


def scenario_on_save_exception_consensus(
    rank: int,
    world_size: int,
    output_dir: str,
    use_cpu: bool = False,
    **_,
) -> None:
    """A rank-0-only on-save exception fails every rank coherently."""
    cfg = TinyConfig()
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        max_steps=1,
        save_steps=1,
        save_strategy="steps",
        report_to=[],
        privacy_noise_multiplier=1.0,
        use_cpu=use_cpu,
        use_compat_patches=False,
    )
    callback_calls = 0

    class _FailOnSavingRank(TrainerCallback):
        def on_save(self, args_, state_, control_, **_kwargs):
            nonlocal callback_calls
            callback_calls += 1
            assert rank == 0
            raise RuntimeError("rank-zero on_save failure")

    trainer = DPTrainer(
        model=TinyForCausalLM(cfg),
        args=args,
        train_dataset=TinyDataset(16, 4, cfg.vocab_size),
        data_collator=_collate,
        callbacks=[_FailOnSavingRank()],
    )
    failed = False
    try:
        trainer.train()
    except OperationError as exc:
        failed = "on_save callback" in str(exc)
    outcomes = [None] * world_size
    dist.all_gather_object(outcomes, (failed, callback_calls))
    assert outcomes == [(True, 1), (True, 0)], outcomes


def scenario_private_step_consensus(
    rank: int,
    world_size: int,
    output_dir: str,
    use_cpu: bool = False,
    **_,
) -> None:
    """Locally consistent but cross-rank step authorities fail closed."""
    cfg = TinyConfig()
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        max_steps=1,
        save_strategy="no",
        report_to=[],
        privacy_noise_multiplier=1.0,
        use_cpu=use_cpu,
        use_compat_patches=False,
    )
    trainer = None

    class _ForgeBothAuthorities(TrainerCallback):
        def on_train_begin(self, args_, state_, control_, **_kwargs):
            if rank == 0:
                state_.global_step = 1
                assert trainer is not None
                assert trainer._ctx is not None
                trainer._ctx.executed_global_step = 1

    trainer = DPTrainer(
        model=TinyForCausalLM(cfg),
        args=args,
        train_dataset=TinyDataset(16, 4, cfg.vocab_size),
        data_collator=_collate,
        callbacks=[_ForgeBothAuthorities()],
    )
    failed = False
    try:
        trainer.train()
    except OperationError as exc:
        failed = "phases diverged" in str(exc)
    outcomes = [None] * world_size
    dist.all_gather_object(outcomes, failed)
    assert outcomes == [True] * world_size, outcomes


def scenario_token_policy_is_frozen(
    rank: int,
    world_size: int,
    output_dir: str,
    use_cpu: bool = False,
    **_,
) -> None:
    """Rank-local callback mutations cannot branch around token reduction."""
    cfg = TinyConfig()
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        max_steps=2,
        save_strategy="no",
        report_to=[],
        privacy_noise_multiplier=1.0,
        include_num_input_tokens_seen="all",
        average_tokens_across_devices=True,
        use_cpu=use_cpu,
        use_compat_patches=False,
    )

    class _MutateLiveTokenArgs(TrainerCallback):
        def on_step_begin(self, args_, state_, control_, **_kwargs):
            if rank == 0:
                args_.include_num_input_tokens_seen = "no"
                args_.average_tokens_across_devices = False

    trainer = DPTrainer(
        model=TinyForCausalLM(cfg),
        args=args,
        train_dataset=TinyDataset(16, 4, cfg.vocab_size),
        data_collator=_collate,
        callbacks=[_MutateLiveTokenArgs()],
    )
    result = trainer.train()
    outcomes = [None] * world_size
    dist.all_gather_object(
        outcomes,
        (result.global_step, trainer.state.num_input_tokens_seen),
    )
    assert outcomes[0] == outcomes[1]
    assert outcomes[0][0] == 2
    assert outcomes[0][1] > 0


def scenario_hub_publication_is_collective_safe(
    rank: int,
    world_size: int,
    output_dir: str,
    use_cpu: bool = False,
    **_,
) -> None:
    """Rank-zero Hub saving never mismatches a DDP barrier/status reduce."""
    cfg = TinyConfig()
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        max_steps=1,
        save_strategy="no",
        report_to=[],
        privacy_noise_multiplier=1.0,
        push_to_hub=True,
        hub_model_id="opaque-tests/ddp-hub-publication",
        use_cpu=use_cpu,
        use_compat_patches=False,
    )

    class _MutateLiveHubPolicy(TrainerCallback):
        def on_train_end(self, args_, state_, control_, **_kwargs):
            if rank == 1:
                args_.push_to_hub = False

    repo = SimpleNamespace(repo_id="opaque-tests/ddp-hub-publication")
    with (
        patch("opaque.api.transformers.trainer._hub._require_hub"),
        patch(
            "opaque.api.transformers.trainer._hub._create_repo",
            return_value=repo,
        ) as create_repo,
        patch("opaque.api.transformers.trainer._hub.create_model_card") as create_card,
        patch(
            "opaque.api.transformers.trainer._hub._upload_folder",
            return_value=SimpleNamespace(),
        ) as upload_folder,
    ):
        trainer = DPTrainer(
            model=TinyForCausalLM(cfg),
            args=args,
            train_dataset=TinyDataset(16, 4, cfg.vocab_size),
            data_collator=_collate,
            callbacks=[_MutateLiveHubPolicy()],
        )
        result = trainer.train()

    outcomes = [None] * world_size
    dist.all_gather_object(
        outcomes,
        (
            result.global_step,
            create_repo.call_count,
            create_card.call_count,
            upload_folder.call_count,
            trainer.args.push_to_hub,
        ),
    )
    assert outcomes == [
        (1, 1, 1, 1, True),
        (1, 0, 0, 0, False),
    ], outcomes


def scenario_env_backend_diagnostic(
    output_dir: str, use_cpu: bool = False, **_
) -> None:
    """Vendor backends are accepted by args but error on unavailable runtime."""
    cfg = TinyConfig()
    model = TinyForCausalLM(cfg)
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        max_steps=1,
        save_strategy="no",
        report_to=[],
        ddp_backend="xccl",
        privacy_noise_multiplier=0.0,
        use_cpu=use_cpu,
        use_compat_patches=False,
    )
    try:
        DPTrainer(
            model=model,
            args=args,
            train_dataset=TinyDataset(8, 4, cfg.vocab_size),
            data_collator=_collate,
        )
    except ValueError as exc:
        msg = str(exc)
        assert "ddp_backend='xccl'" in msg or 'ddp_backend="xccl"' in msg
        return
    raise AssertionError("Expected DPTrainer to fail fast for unavailable xccl runtime")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


SCENARIOS = {
    "runtime_foundation": scenario_runtime_foundation,
    "per_rank_partition": scenario_per_rank_partition,
    "eval_gather": scenario_eval_gather,
    "eval_gather_empty_rank": scenario_eval_gather_empty_rank,
    "batch_eval_metrics": scenario_batch_eval_metrics,
    "rank_gating_and_worker_seed": scenario_rank_gating_and_worker_seed,
    "gather_paths": scenario_gather_paths,
    "step_authority_consensus": scenario_step_authority_consensus,
    "post_optimizer_oom_consensus": scenario_post_optimizer_oom_consensus,
    "input_oom_retry_consensus": scenario_input_oom_retry_consensus,
    "metric_oom_consensus": scenario_metric_oom_consensus,
    "token_reduce_prep_oom_consensus": scenario_token_reduce_prep_oom_consensus,
    "on_save_exception_consensus": scenario_on_save_exception_consensus,
    "private_step_consensus": scenario_private_step_consensus,
    "token_policy_is_frozen": scenario_token_policy_is_frozen,
    "hub_publication_is_collective_safe": scenario_hub_publication_is_collective_safe,
    "env_backend_diagnostic": scenario_env_backend_diagnostic,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--scenario", required=True, choices=list(SCENARIOS))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--backend", default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or tempfile.mkdtemp(prefix="dpt_ddp_")
    device = _setup_ddp(args.rank, args.world_size, args.port, backend=args.backend)
    try:
        # `_setup_ddp` owns backend resolution, so scenarios receive the
        # placement it decided rather than re-deriving it from the raw
        # `--backend` argument, which is unset whenever the backend is being
        # auto-selected.  `_setup_ddp` also exports `LOCAL_RANK`, so a scenario
        # that builds `TrainingArguments` without this flag puts rank 1 on
        # `cuda:1`.
        SCENARIOS[args.scenario](
            rank=args.rank,
            world_size=args.world_size,
            output_dir=output_dir,
            use_cpu=device.type == "cpu",
        )
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        import traceback

        print(
            json.dumps(
                {
                    "rank": int(os.environ.get("RANK", -1)),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            ),
            file=sys.stderr,
        )
        sys.exit(1)
