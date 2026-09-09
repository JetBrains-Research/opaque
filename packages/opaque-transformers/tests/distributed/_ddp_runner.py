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
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

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
# Router-load release (T12) and rank-local sampler resume (T25) scenarios
# ---------------------------------------------------------------------------


class RecordingDataset(TinyDataset):
    """``TinyDataset`` that logs every global index handed to ``__getitem__``.

    Under DDP the trainer wraps the dataset in ``Subset`` / ``local_shard``
    views, which forward each access to this base object with the *global*
    index, so the log is a per-rank record of the Poisson inclusion draws.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accesses: list[int] = []

    def __getitem__(self, idx: int):
        self.accesses.append(int(idx))
        return super().__getitem__(idx)


class _BatchRecorder:
    """Cut the dataset access log into per-step inclusion sets.

    The train loader (and its one-row collator priming) is built before
    ``on_epoch_begin`` fires, and batch ``t`` is fetched right before
    ``on_step_begin`` of step ``t``, so clearing at ``on_epoch_begin`` and
    cutting at ``on_step_begin`` yields exactly one sorted index list per
    optimizer step (an empty list for an empty Poisson round).  At
    ``on_train_end`` the live sampler snapshot is captured while the
    training context still exists.
    """

    def __init__(self, dataset: RecordingDataset) -> None:
        from transformers.trainer_callback import TrainerCallback

        recorder = self
        dataset.accesses.clear()
        self.dataset = dataset
        self.batches: list[list[int]] = []
        self.sampler_state: dict | None = None

        class _Callback(TrainerCallback):
            def __init__(self, trainer) -> None:
                self.trainer = trainer

            def on_epoch_begin(self, args, state, control, **kwargs):
                recorder.dataset.accesses.clear()
                return control

            def on_step_begin(self, args, state, control, **kwargs):
                recorder.batches.append(sorted(recorder.dataset.accesses))
                recorder.dataset.accesses.clear()
                return control

            def on_train_end(self, args, state, control, **kwargs):
                from opaque.serialization import state_dict

                sampler = self.trainer._ctx.current_sampler
                recorder.sampler_state = dict(state_dict(sampler))
                return control

        self._callback_cls = _Callback

    def attach(self, trainer) -> None:
        trainer.add_callback(self._callback_cls(trainer))


def _resume_ranks_args(
    output_dir: str, use_cpu: bool, **overrides
) -> TrainingArguments:
    defaults = {
        "output_dir": output_dir,
        "per_device_train_batch_size": 4,
        "max_steps": 8,
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": 3,
        "seed": 7,
        "privacy_noise_multiplier": 1.0,
        "clipping_norm": 1.0,
        "report_to": [],
        "disable_tqdm": True,
        "use_cpu": use_cpu,
        "use_compat_patches": False,
    }
    defaults.update(overrides)
    return TrainingArguments(**defaults)


def scenario_resume_ranks(
    rank: int, world_size: int, output_dir: str, use_cpu: bool = False, **_
) -> None:
    """Verify each rank resumes its *own* Poisson stream after a checkpoint.

    A continuous 8-step run saves ``checkpoint-3`` (sampler snapshot written
    by rank 0); a second trainer resumes from it for the remaining 5 steps.
    Every rank's post-resume inclusion masks must equal steps 4..8 of its
    own continuous run, the two ranks' masks must differ (independent
    per-rank coins, which the amplification argument needs), and the
    restored sampler must carry this rank's own rank-folded stream key
    rather than the rank-0 key stored in the snapshot.
    """
    steps_before = 3
    total_steps = 8
    cfg = TinyConfig()

    def _run(out_dir: str, resume: str | None) -> tuple[list[list[int]], dict]:
        torch.manual_seed(0)
        model = TinyForCausalLM(cfg)
        ds = RecordingDataset(n=64, seq_len=8, vocab=cfg.vocab_size)
        recorder = _BatchRecorder(ds)
        args = _resume_ranks_args(out_dir, use_cpu, resume_from_checkpoint=resume)
        trainer = DPTrainer(
            model=model, args=args, train_dataset=ds, data_collator=_collate
        )
        assert trainer._ddp.world_size == world_size
        recorder.attach(trainer)
        trainer.train()
        assert recorder.sampler_state is not None
        return recorder.batches, recorder.sampler_state

    continuous_dir = str(Path(output_dir) / "continuous")
    continuous_batches, continuous_sampler = _run(continuous_dir, None)
    assert len(continuous_batches) == total_steps, continuous_batches
    assert continuous_sampler["consumed"] == total_steps
    ckpt_dir = Path(continuous_dir) / f"checkpoint-{steps_before}"
    # Rank 0 publishes the checkpoint directory; wait for it before resuming.
    dist.barrier()
    assert ckpt_dir.is_dir(), sorted(Path(continuous_dir).iterdir())
    assert (ckpt_dir / f"rng_state_{rank}.pth").exists()

    resumed_batches, resumed_sampler = _run(
        str(Path(output_dir) / "resumed"), str(ckpt_dir)
    )
    assert len(resumed_batches) == total_steps - steps_before, resumed_batches
    assert resumed_sampler["consumed"] == total_steps

    # Each rank continues its own pre-checkpoint stream bit for bit.
    assert resumed_batches == continuous_batches[steps_before:], (
        f"rank {rank}: resumed {resumed_batches} != "
        f"continuous tail {continuous_batches[steps_before:]}"
    )
    assert resumed_sampler["key_seed"] == continuous_sampler["key_seed"]
    assert resumed_sampler == continuous_sampler

    # Cross-rank: the streams are distinct (rank-folded keys), so the masks
    # differ, and the snapshot written by rank 0 was re-keyed on rank 1.
    gathered: list = [None] * world_size
    dist.all_gather_object(
        gathered,
        {
            "after": resumed_batches,
            "key_seed": resumed_sampler["key_seed"],
            "continuous_key_seed": continuous_sampler["key_seed"],
        },
    )
    seeds = {g["key_seed"] for g in gathered}
    assert len(seeds) == world_size, f"ranks share a sampler key: {gathered}"
    for r in range(1, world_size):
        assert gathered[r]["after"] != gathered[0]["after"], gathered
        assert gathered[r]["key_seed"] != gathered[0]["continuous_key_seed"]
    # Shards are disjoint: no rank ever draws another rank's records.
    shard = 64 // world_size
    for r, g in enumerate(gathered):
        for batch in g["after"]:
            assert all(r * shard <= i < (r + 1) * shard for i in batch), (r, batch)
    # Sanity: the comparison above is not vacuous (some non-empty rounds).
    assert any(batch for g in gathered for batch in g["after"])


_MELLUM_NUM_EXPERTS = 8
_MELLUM_TOP_K = 2
_MELLUM_NUM_LAYERS = 2
_MELLUM_T_MAX = 16
_MELLUM_VOCAB = 128
_MELLUM_TRAINABLE = ("q_proj", "k_proj", "v_proj", "o_proj", ".mlp.gate.")


def _tiny_mellum():
    from transformers import MellumConfig, MellumForCausalLM

    torch.manual_seed(0)
    config = MellumConfig(
        vocab_size=_MELLUM_VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=_MELLUM_NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_theta=10000.0,
        num_experts=_MELLUM_NUM_EXPERTS,
        num_experts_per_tok=_MELLUM_TOP_K,
        moe_intermediate_size=32,
        router_aux_loss_coef=0.01,
    )
    model = MellumForCausalLM(config)
    for name, p in model.named_parameters():
        p.requires_grad_(any(s in name for s in _MELLUM_TRAINABLE))
    return model.train()


class RaggedDataset(Dataset):
    """Right-padded ragged rows with an attention mask and ``-100`` labels."""

    def __init__(self, n: int = 32, seed: int = 1) -> None:
        g = torch.Generator().manual_seed(seed)
        ids = torch.randint(3, _MELLUM_VOCAB, (n, _MELLUM_T_MAX), generator=g)
        lengths = torch.randint(
            _MELLUM_T_MAX // 2, _MELLUM_T_MAX + 1, (n,), generator=g
        )
        mask = (torch.arange(_MELLUM_T_MAX)[None] < lengths[:, None]).long()
        self.input_ids = torch.where(mask.bool(), ids, torch.zeros_like(ids))
        self.attention_mask = mask
        self.labels = torch.where(mask.bool(), ids, torch.full_like(ids, -100))

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels": self.labels[i],
        }


def _collate_ragged(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if len(rows) == 0:
        return {
            "input_ids": torch.zeros((0, _MELLUM_T_MAX), dtype=torch.long),
            "attention_mask": torch.zeros((0, _MELLUM_T_MAX), dtype=torch.long),
            "labels": torch.zeros((0, _MELLUM_T_MAX), dtype=torch.long),
        }
    return {k: torch.stack([r[k] for r in rows]) for k in rows[0]}


def _state_payload(state) -> dict[str, Any]:
    """Plain ``dict`` of a ``RouterLoadState`` (tensors on CPU) for gathering."""
    import dataclasses

    out = {}
    for field in dataclasses.fields(state):
        value = getattr(state, field.name)
        out[field.name] = value.detach().cpu() if torch.is_tensor(value) else value
    return out


def _assert_same_payload(a: dict[str, Any], b: dict[str, Any], what: str) -> None:
    assert a.keys() == b.keys(), (what, sorted(a), sorted(b))
    for name in a:
        x, y = a[name], b[name]
        if torch.is_tensor(x):
            assert torch.is_tensor(y), (what, name, x, y)
            assert torch.equal(x, y), (what, name, x, y)
        else:
            assert x == y, (what, name, x, y)


def scenario_router_load_release(
    rank: int, world_size: int, output_dir: str, use_cpu: bool = False, **_
) -> None:
    """Verify the router-load release is rank-identical under DDP.

    Both ranks train a tiny Mellum for three steps with
    ``router_load_release="monitor"``.  The probe leaf is part of the
    clipped pytree that ``sum_gradients_`` all-reduces and the noise key is
    shared, so the *noised* probe leaf every rank hands to the release
    callback must be bit-identical across ranks even though the ranks
    clip disjoint local batches; consequently the public post-processing
    state (``RouterLoadState`` / ``f_tilde``) is identical too, and the
    sidecar rank 0 writes equals the state rank 1 holds.
    """
    from transformers.trainer_callback import TrainerCallback

    from opaque.api.transformers.moe_load import PROBE_NAME
    from opaque.api.transformers.trainer._router_load import (
        ROUTER_LOAD_STATE_NAME,
        RouterLoadCallback,
    )
    from opaque.serialization import from_state_dict

    steps = 3
    model = _tiny_mellum()
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        max_steps=steps,
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="steps",
        save_steps=steps,
        report_to=[],
        disable_tqdm=True,
        use_cpu=use_cpu,
        seed=0,
        privacy_noise_multiplier=1.0,
        clipping_norm=1.0,
        learning_rate=1e-3,
        optim="adamw",
        router_load_release="monitor",
        router_load_max_tokens=_MELLUM_T_MAX,
    )
    ds = RaggedDataset()
    trainer = DPTrainer(
        model=model, args=args, train_dataset=ds, data_collator=_collate_ragged
    )
    assert trainer._ddp.world_size == world_size

    probes: list[torch.Tensor] = []
    states: list[dict[str, Any]] = []

    class _Recorder(TrainerCallback):
        # Registered before ``train()``: the trainer appends its own
        # ``RouterLoadCallback`` in ``_setup_training``, so this hook sees the
        # noised probe leaf *before* the release consumes and zeroes it.
        def on_pre_optimizer_step(self, args, state, control, grads=None, **kw):
            probes.append(grads.pytree[PROBE_NAME].detach().cpu().clone())
            return control

        def on_optimizer_step(self, args, state, control, trainable_params=None, **kw):
            assert not trainable_params[PROBE_NAME].any(), "probe drifted"
            states.append(_state_payload(trainer._router_load.state))
            return control

    trainer.add_callback(_Recorder())
    trainer.train()
    assert len(probes) == steps
    assert len(states) == steps
    assert all(p.shape == (_MELLUM_NUM_LAYERS, _MELLUM_NUM_EXPERTS) for p in probes)
    # The hook ordering above held: the release was still in the leaf.
    assert any(p.any() for p in probes), "probe leaf already zeroed on capture"
    assert states[-1]["step"] == steps
    assert states[-1]["num_experts"] == _MELLUM_NUM_EXPERTS
    f_tilde = states[-1]["f_tilde"]
    assert f_tilde.min() >= 0
    assert f_tilde.max() <= 1
    assert torch.allclose(f_tilde.sum(), torch.tensor(float(_MELLUM_TOP_K)), atol=1e-5)

    gathered: list = [None] * world_size
    dist.all_gather_object(gathered, {"probes": probes, "states": states})
    for r in range(1, world_size):
        for t in range(steps):
            assert torch.equal(gathered[r]["probes"][t], gathered[0]["probes"][t]), (
                f"step {t}: noised probe leaf differs between rank 0 and rank {r}"
            )
            _assert_same_payload(
                gathered[r]["states"][t], gathered[0]["states"][t], f"state step {t}"
            )

    # The sidecar rank 0 wrote at step 3 is the state every rank holds.
    dist.barrier()
    sidecar = Path(output_dir) / f"checkpoint-{steps}" / ROUTER_LOAD_STATE_NAME
    assert sidecar.exists(), sorted(Path(output_dir).iterdir())
    payload = torch.load(str(sidecar), map_location="cpu", weights_only=False)
    # ``train()`` clears ``trainer._router_load``; the release callback stays
    # registered and still holds this rank's final state as the template.
    release = next(
        cb
        for cb in trainer.callback_handler.callbacks
        if isinstance(cb, RouterLoadCallback)
    )
    restored = from_state_dict(release.state, payload["state"])
    _assert_same_payload(_state_payload(restored), states[-1], "sidecar")
    _assert_same_payload(_state_payload(release.state), states[-1], "callback")


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
    "env_backend_diagnostic": scenario_env_backend_diagnostic,
    "resume_ranks": scenario_resume_ranks,
    "router_load_release": scenario_router_load_release,
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
