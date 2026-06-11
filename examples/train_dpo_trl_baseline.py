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
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET = "trl-lib/ultrafeedback_binarized"
NUM_TRAIN_SAMPLES = int(os.environ.get("NUM_TRAIN_SAMPLES", "2000"))
NUM_EVAL_SAMPLES = int(os.environ.get("NUM_EVAL_SAMPLES", "0"))
MAX_LENGTH = 512
# Effective batch = PER_DEVICE_BATCH * GRAD_ACCUM, env-configurable so the
# baseline can match the opaque ``--batch-size`` (e.g. BATCH_SIZE=32 ⇒ 2*16).
PER_DEVICE_BATCH = 2
GRAD_ACCUM = max(1, int(os.environ.get("BATCH_SIZE", "8")) // PER_DEVICE_BATCH)
LR = float(os.environ.get("LR", "1e-4"))
BETA = 0.1
NUM_STEPS = int(os.environ.get("NUM_STEPS", "50"))
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", "0")) or None
# WITH_REPLACEMENT=1 → iid with-replacement train sampling (no epochs), matching
# the opaque trainer's Poisson sampler. Removes the epoch-boundary memorization
# STEP that without-replacement epoch shuffling produces (the step's size scales
# with memorization, but its existence is purely the fresh→seen epoch boundary).
WITH_REPLACEMENT = os.environ.get("WITH_REPLACEMENT", "0") == "1"


class _NoEpochDPOTrainer(DPOTrainer):
    """DPO with iid WITH-REPLACEMENT train sampling — one ``num_samples`` =
    NUM_STEPS*batch "epoch" covers the whole run, so there are NO epoch
    boundaries and thus no epoch-boundary memorization step. Matches the opaque
    trainer's Poisson sampler (which DP requires), making the train-loss curves
    shape-comparable. Only the sampler changes; the model/loss are untouched."""

    def _get_train_sampler(self, *args, **kwargs):
        from torch.utils.data import RandomSampler

        n = NUM_STEPS * PER_DEVICE_BATCH * GRAD_ACCUM
        return RandomSampler(self.train_dataset, replacement=True, num_samples=n)


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

    # USE_LORA=1 → LoRA-DPO (rank-constrained, resists memorizing the 10k-pair
    # train set; PEFT null-ref = frozen base with the adapter disabled, so no
    # separate ref_model). Default 0 = full fine-tuning baseline.
    use_lora = os.environ.get("USE_LORA", "0") == "1"
    if use_lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=int(os.environ.get("LORA_R", "8")),
            lora_alpha=int(os.environ.get("LORA_ALPHA", "16")),
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.0,
        )
        ref_model = None
    else:
        peft_config = None
        ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)  # full-FT

    raw = load_dataset(DATASET, split="train", streaming=True)
    take_total = NUM_TRAIN_SAMPLES + (NUM_EVAL_SAMPLES if EVAL_STEPS else 0)
    all_rows = [row for _, row in zip(range(take_total), raw)]
    train_dataset = Dataset.from_list(all_rows[:NUM_TRAIN_SAMPLES])
    eval_dataset = (
        Dataset.from_list(all_rows[NUM_TRAIN_SAMPLES:])
        if EVAL_STEPS
        else None
    )

    eval_kwargs: dict = {}
    if EVAL_STEPS:
        eval_kwargs = {
            "eval_strategy": "steps",
            "eval_steps": EVAL_STEPS,
            "per_device_eval_batch_size": PER_DEVICE_BATCH,
        }

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
        # Native HF W&B integration: it reuses the run opened by wandb.init()
        # above and logs train/* and eval/* under the SAME keys the opaque
        # DPOTrainer uses (eval/rewards/margins, …), so the noDP/DP/baseline
        # triple overlays on one panel. No custom callback needed.
        report_to=["wandb"],
        optim="adamw_torch",
        **eval_kwargs,
    )

    trainer_cls = _NoEpochDPOTrainer if WITH_REPLACEMENT else DPOTrainer
    trainer = trainer_cls(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    out = trainer.train()
    print("Training complete:", out)
    if hasattr(out, "metrics"):
        wandb.summary["final_train_loss"] = out.metrics.get("train_loss")
    wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
