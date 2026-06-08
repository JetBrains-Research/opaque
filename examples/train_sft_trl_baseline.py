"""Upstream-TRL SFTTrainer baseline (no opaque imports).

V5b validation script: trains the same Qwen/Qwen2.5-0.5B + roneneldan/TinyStories
slice (2000 samples) with upstream ``trl.SFTTrainer`` so its per-step train/loss
can be compared against the opaque ``SFTTrainer`` at noise=0/clipping=1e9.

Logs train/loss to W&B at every step. No DP — this is the upstream baseline.
"""
from __future__ import annotations

import os
import sys

import torch
import wandb
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer


SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATASET = "roneneldan/TinyStories"
NUM_TRAIN_SAMPLES = 2000
MAX_LENGTH = 512
BATCH_SIZE = 8
LR = 1e-4
NUM_STEPS = 50
LORA_R = 8
LORA_ALPHA = 16
LORA_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class WandbStepLossCallback(TrainerCallback):
    """Mirror trainer ``logs`` (which include ``loss``) to W&B at every step."""

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
    run_name = os.environ.get("RUN_NAME", "trl-sft-baseline")
    wandb.init(
        entity="federated-compute",
        project="opaque",
        name=run_name,
        tags=["val/trl-trainers-r1b"],
        config={
            "scenario": "V5b-sft-trl-baseline",
            "model": MODEL_NAME,
            "dataset": DATASET,
            "num_train_samples": NUM_TRAIN_SAMPLES,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "num_steps": NUM_STEPS,
            "seed": SEED,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_modules": LORA_MODULES,
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_MODULES,
        lora_dropout=0.0,
    )

    raw = load_dataset(DATASET, split="train", streaming=True)
    rows = [row for _, row in zip(range(NUM_TRAIN_SAMPLES), raw)]
    train_dataset = Dataset.from_list(rows)

    sft_args = SFTConfig(
        output_dir="trainer_output/sft_trl_baseline",
        overwrite_output_dir=True,
        dataset_text_field="text",
        max_length=MAX_LENGTH,
        per_device_train_batch_size=BATCH_SIZE,
        max_steps=NUM_STEPS,
        learning_rate=LR,
        logging_steps=1,
        save_strategy="no",
        seed=SEED,
        use_cpu=not torch.cuda.is_available(),
        bf16=torch.cuda.is_available(),
        report_to=[],  # we log to wandb manually via the callback so opaque/baseline
                      # share the same entity/project/name plumbing
        optim="adamw_torch",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[WandbStepLossCallback()],
    )
    out = trainer.train()
    print("Training complete:", out)
    final_loss = out.metrics.get("train_loss") if hasattr(out, "metrics") else None
    if final_loss is not None:
        wandb.summary["final_train_loss"] = final_loss
    wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
