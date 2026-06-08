"""Upstream-TRL DPOTrainer baseline (no opaque imports).

V5b validation script: trains the same Qwen/Qwen2.5-0.5B-Instruct +
trl-lib/ultrafeedback_binarized slice with upstream ``trl.DPOTrainer`` (sigmoid,
beta=0.1) so its per-step train/loss can be compared against the opaque
``DPOTrainer`` at noise=0/clipping=1e9.

Logs train/loss + rewards/* metrics to W&B at every step. Full fine-tuning
(no LoRA), matching the opaque DPO run (which doesn't pass --peft).
"""
from __future__ import annotations

import os

import torch
import wandb
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer


SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET = "trl-lib/ultrafeedback_binarized"
NUM_TRAIN_SAMPLES = 2000
MAX_LENGTH = 512
PER_DEVICE_BATCH = 2
GRAD_ACCUM = 4  # effective batch 8 — matches opaque batch_size=8
LR = 1e-4
BETA = 0.1
NUM_STEPS = 50


class WandbStepLossCallback(TrainerCallback):
    """Mirror trainer ``logs`` (which include ``loss``, ``rewards/*``) to W&B."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not wandb.run:
            return
        payload = {f"train/{k}": v for k, v in logs.items() if isinstance(v, (int, float))}
        if "loss" in logs:
            payload["train/loss"] = logs["loss"]
        if payload:
            payload["train/global_step"] = state.global_step
            wandb.log(payload, step=state.global_step)


def main() -> int:
    run_name = os.environ.get("RUN_NAME", "trl-dpo-baseline")
    wandb.init(
        entity="federated-compute",
        project="opaque",
        name=run_name,
        tags=["val/trl-trainers-r1b"],
        config={
            "scenario": "V5b-dpo-trl-baseline",
            "model": MODEL_NAME,
            "dataset": DATASET,
            "num_train_samples": NUM_TRAIN_SAMPLES,
            "max_length": MAX_LENGTH,
            "per_device_batch": PER_DEVICE_BATCH,
            "grad_accum": GRAD_ACCUM,
            "lr": LR,
            "beta": BETA,
            "num_steps": NUM_STEPS,
            "seed": SEED,
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)  # full-FT baseline

    raw = load_dataset(DATASET, split="train", streaming=True)
    rows = [row for _, row in zip(range(NUM_TRAIN_SAMPLES), raw)]
    train_dataset = Dataset.from_list(rows)

    dpo_args = DPOConfig(
        output_dir="trainer_output/dpo_trl_baseline",
        overwrite_output_dir=True,
        beta=BETA,
        loss_type="sigmoid",
        max_length=MAX_LENGTH,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=NUM_STEPS,
        learning_rate=LR,
        logging_steps=1,
        save_strategy="no",
        seed=SEED,
        use_cpu=not torch.cuda.is_available(),
        bf16=torch.cuda.is_available(),
        report_to=[],
        optim="adamw_torch",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[WandbStepLossCallback()],
    )
    out = trainer.train()
    print("Training complete:", out)
    if hasattr(out, "metrics"):
        wandb.summary["final_train_loss"] = out.metrics.get("train_loss")
    wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
