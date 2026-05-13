"""Multi-GPU smoke runner for DPTrainer DDP and HPO paths.

Usage:
  # DDP train/eval smoke
  uv run torchrun --nproc-per-node=2 \
    packages/opaque-transformers/tests/distributed/smoke_ddp_hpo.py \
    --mode ddp --output-dir /tmp/opaque-smoke-ddp

  # Distributed Optuna HPO smoke
  uv run torchrun --nproc-per-node=2 \
    packages/opaque-transformers/tests/distributed/smoke_ddp_hpo.py \
    --mode hpo --output-dir /tmp/opaque-smoke-hpo

  # Ray Tune smoke (controller rank only under torchrun)
  uv run torchrun --nproc-per-node=2 \
    packages/opaque-transformers/tests/distributed/smoke_ddp_hpo.py \
    --mode ray --output-dir /tmp/opaque-smoke-ray

  # W&B sweep smoke (controller rank only under torchrun, offline mode)
  uv run torchrun --nproc-per-node=2 \
    packages/opaque-transformers/tests/distributed/smoke_ddp_hpo.py \
    --mode wandb --output-dir /tmp/opaque-smoke-wandb
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput

from opaque.transformers.trainer import DPTrainer, TrainingArguments


class TinyConfig(PretrainedConfig):
    model_type = "tiny_smoke"

    def __init__(self, hidden_size: int = 8, num_labels: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_labels = num_labels


class TinyClassifier(PreTrainedModel):
    config_class = TinyConfig
    main_input_name = "x"

    def __init__(self, config: TinyConfig):
        super().__init__(config)
        self.proj = torch.nn.Linear(4, config.hidden_size)
        self.head = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x, labels=None, **_):
        logits = self.head(torch.relu(self.proj(x)))
        loss = None
        if labels is not None:
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
                labels = labels.reshape(1)
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


class TinyDataset(Dataset):
    def __init__(self):
        self._samples = [
            {"x": torch.tensor([1.0, 0.0, 0.0, 0.0]), "labels": torch.tensor(0)},
            {"x": torch.tensor([0.0, 1.0, 0.0, 0.0]), "labels": torch.tensor(1)},
            {"x": torch.tensor([0.0, 0.0, 1.0, 0.0]), "labels": torch.tensor(0)},
            {"x": torch.tensor([0.0, 0.0, 0.0, 1.0]), "labels": torch.tensor(1)},
            {"x": torch.tensor([1.0, 1.0, 0.0, 0.0]), "labels": torch.tensor(0)},
            {"x": torch.tensor([0.0, 1.0, 1.0, 0.0]), "labels": torch.tensor(1)},
            {"x": torch.tensor([0.0, 0.0, 1.0, 1.0]), "labels": torch.tensor(0)},
            {"x": torch.tensor([1.0, 0.0, 0.0, 1.0]), "labels": torch.tensor(1)},
        ]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        return self._samples[idx]


def _setup_dist() -> tuple[int, int]:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, world_size


def _build_args(output_dir: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        max_steps=2,
        eval_strategy="steps",
        eval_steps=1,
        save_strategy="no",
        report_to=[],
        seed=123,
        privacy_target_epsilon=10.0,
        privacy_noise_multiplier=1.0,
    )


def run_ddp_smoke(output_dir: str, rank: int) -> None:
    args = _build_args(output_dir)
    trainer = DPTrainer(
        model=TinyClassifier(TinyConfig()),
        args=args,
        train_dataset=TinyDataset(),
        eval_dataset=TinyDataset(),
    )
    train_out = trainer.train()
    assert train_out.global_step >= 1
    metrics = trainer.evaluate()
    if rank == 0:
        print(f"[smoke-ddp] eval keys={sorted(metrics.keys())}")


def run_hpo_smoke(output_dir: str, rank: int) -> None:
    import optuna

    def model_init(trial=None):
        del trial
        return TinyClassifier(TinyConfig())

    args = _build_args(output_dir)
    trainer = DPTrainer(
        model_init=model_init,
        args=args,
        train_dataset=TinyDataset(),
        eval_dataset=TinyDataset(),
    )

    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_categorical("learning_rate", [1e-3, 5e-4])
        }

    best = trainer.hyperparameter_search(
        backend="optuna",
        hp_space=hp_space,
        compute_objective=lambda metrics: metrics["eval_loss"],
        n_trials=2,
        direction="minimize",
        sampler=optuna.samplers.GridSampler({"learning_rate": [1e-3, 5e-4]}),
    )
    if rank == 0:
        assert best is not None
        print(f"[smoke-hpo] best_run={best.run_id} obj={best.objective}")
    else:
        assert best is None


def run_ray_smoke(output_dir: str, rank: int) -> None:
    # HF-style usage expects one process to control the Ray sweep.
    if rank != 0:
        return
    import ray.tune as tune

    def model_init(trial=None):
        del trial
        return TinyClassifier(TinyConfig())

    args = _build_args(output_dir)
    trainer = DPTrainer(
        model_init=model_init,
        args=args,
        train_dataset=TinyDataset(),
        eval_dataset=TinyDataset(),
    )

    best = trainer.hyperparameter_search(
        backend="ray",
        hp_space=lambda _t: {"learning_rate": tune.loguniform(5e-4, 1e-3)},
        compute_objective=lambda metrics: metrics["eval_loss"],
        n_trials=2,
        direction="minimize",
        resources_per_trial={"cpu": 1, "gpu": 1 if torch.cuda.is_available() else 0},
        storage_path=str(Path(output_dir) / "ray-storage"),
    )
    assert best is not None
    print(f"[smoke-ray] best_run={best.run_id} obj={best.objective}")


def run_wandb_smoke(output_dir: str, rank: int) -> None:
    # HF-style usage expects one process to control the W&B sweep agent.
    if rank != 0:
        return
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_SILENT", "true")
    if not os.environ.get("WANDB_API_KEY"):
        print("[smoke-wandb] skipped: WANDB_API_KEY is not set")
        return

    def model_init(trial=None):
        del trial
        return TinyClassifier(TinyConfig())

    args = _build_args(output_dir)
    trainer = DPTrainer(
        model_init=model_init,
        args=args,
        train_dataset=TinyDataset(),
        eval_dataset=TinyDataset(),
    )

    best = trainer.hyperparameter_search(
        backend="wandb",
        n_trials=2,
        direction="minimize",
        hp_space=lambda _t: {
            "method": "grid",
            "metric": {"name": "objective"},
            "parameters": {"learning_rate": {"values": [5e-4, 1e-3]}},
        },
        compute_objective=lambda metrics: metrics["eval_loss"],
        project="opaque-smoke",
        metric="eval/loss",
    )
    assert best is not None
    print(f"[smoke-wandb] best_run={best.run_id} obj={best.objective}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ddp", "hpo", "ray", "wandb"], required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    rank, _world_size = _setup_dist()
    try:
        if args.mode == "ddp":
            run_ddp_smoke(str(output_dir), rank)
        elif args.mode == "hpo":
            run_hpo_smoke(str(output_dir), rank)
        elif args.mode == "ray":
            run_ray_smoke(str(output_dir), rank)
        else:
            run_wandb_smoke(str(output_dir), rank)
        dist.barrier()
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
