"""Upstream-TRL DPOTrainer baseline (no opaque imports).

Trains the same JetBrains/Mellum-4b-base + trl-lib/ultrafeedback_binarized
slice with upstream ``trl.DPOTrainer`` (sigmoid, beta=0.1) so its per-step
train/loss can be compared against the opaque ``DPOTrainer`` at
noise=0/clipping=1e9 (LoRA, matching the opaque Mellum DPO preset).

Mellum-4b-base has no chat template; we install a minimal ChatML template at
tokenizer load time so ``apply_chat_template`` resolves the list-of-message
preference rows the same way the opaque trainer does.

Logs train/loss + rewards/* metrics to W&B at every step. Defaults to LoRA
fine-tuning (USE_LORA=1) to match the opaque ``--peft`` runs.
"""
from __future__ import annotations

import os

import torch
import wandb
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


SEED = 42
MODEL_NAME = os.environ.get("MODEL_NAME", "JetBrains/Mellum2-12B-A2.5B-Base")
DATASET = os.environ.get("DATASET", "zed-industries/zeta")
# Zeta DPO split has 132 pairs; cap to it so we don't loop the iterator.
NUM_TRAIN_SAMPLES = int(os.environ.get("NUM_TRAIN_SAMPLES", "132"))
NUM_EVAL_SAMPLES = int(os.environ.get("NUM_EVAL_SAMPLES", "0"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "1024"))
# Effective batch = PER_DEVICE_BATCH * GRAD_ACCUM, env-configurable so the
# baseline can match the opaque ``--batch-size`` (BATCH_SIZE=16 with the
# Mellum2 default PER_DEVICE_BATCH=2 ⇒ 8 grad-accum steps).
PER_DEVICE_BATCH = int(os.environ.get("PER_DEVICE_BATCH", "2"))
GRAD_ACCUM = max(1, int(os.environ.get("BATCH_SIZE", "16")) // PER_DEVICE_BATCH)
LR = float(os.environ.get("LR", "5e-5"))
BETA = float(os.environ.get("BETA", "0.1"))
NUM_STEPS = int(os.environ.get("NUM_STEPS", "100"))
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", "0")) or None
# WITH_REPLACEMENT=1 → iid with-replacement train sampling (no epochs), matching
# the opaque trainer's Poisson sampler. Removes the epoch-boundary memorization
# STEP that without-replacement epoch shuffling produces (the step's size scales
# with memorization, but its existence is purely the fresh→seen epoch boundary).
WITH_REPLACEMENT = os.environ.get("WITH_REPLACEMENT", "0") == "1"


def _to_trl_canonical_dpo(row: dict) -> dict:
    """Collapse non-TRL preference shapes into (prompt, chosen, rejected).

    Matches the opaque trainer's canonicalization so the upstream-TRL baseline
    sees the same prompt for each preference pair under either CyberNative
    (system, question) or zed-industries/zeta NES (events, input, output)
    shape.
    """
    if "output" in row and "input" in row and "rejected" in row:
        events = (row.get("events") or "").strip()
        input_code = row["input"]
        prompt = f"{events}\n\n{input_code}" if events else input_code
        return {
            "prompt": prompt,
            "chosen": row["output"],
            "rejected": row["rejected"],
        }
    system = (row.get("system") or "").strip()
    question = row["question"]
    return {
        "prompt": f"{system}\n\n{question}" if system else question,
        "chosen": row["chosen"],
        "rejected": row["rejected"],
    }


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
    if tokenizer.chat_template is None:
        # Mellum-4b-base ships no chat template; install ChatML so the upstream
        # TRL DPOTrainer's apply_chat_template resolves the list-of-message
        # ultrafeedback rows the same way the opaque trainer does.
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # USE_LORA=1 (default) → LoRA-DPO matching the opaque mellum-ultrafeedback
    # preset (rank-constrained, resists memorizing the 10k-pair train set;
    # PEFT null-ref = frozen base with the adapter disabled, so no separate
    # ref_model). USE_LORA=0 = full fine-tuning baseline.
    use_lora = os.environ.get("USE_LORA", "1") == "1"
    if use_lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=int(os.environ.get("LORA_R", "16")),
            lora_alpha=int(os.environ.get("LORA_ALPHA", "32")),
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

    dataset_split = os.environ.get("DATASET_SPLIT", "train")
    raw = load_dataset(DATASET, split=dataset_split, streaming=True)
    take_total = NUM_TRAIN_SAMPLES + (NUM_EVAL_SAMPLES if EVAL_STEPS else 0)
    all_rows = [row for _, row in zip(range(take_total), raw)]
    # Canonicalize non-TRL-canonical preference shapes (CyberNative,
    # zed-industries/zeta) to (prompt, chosen, rejected) so upstream TRL
    # DPOTrainer sees the same row schema as the opaque trainer.
    if all_rows and "prompt" not in all_rows[0] and (
        "question" in all_rows[0] or "input" in all_rows[0]
    ):
        all_rows = [_to_trl_canonical_dpo(row) for row in all_rows]
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
