"""DP Direct Preference Optimization via the class-based ``DPOTrainer``.

Trainer-based sibling of the functional ``examples/train_dpo.py``. The
functional script wires ``opaque.alignment`` DPO primitives into a hand-rolled
DP-SGD loop; this one hands the same primitives to
:class:`opaque.transformers.trl.DPOTrainer`, which precomputes the reference
log-probs and orchestrates the per-example DP path on top of ``DPTrainer``.

Example::

    uv run python examples/train_dpo_trainer.py \\
      --model Qwen/Qwen2.5-0.5B-Instruct \\
      --dataset trl-lib/ultrafeedback_binarized \\
      --beta 0.1 --max-length 512 --batch-size 8 --microbatch-size 2 \\
      --max-steps 50 --learning-rate 1e-4 \\
      --clipping-norm 1.0 --noise-multiplier 0.8 --log-steps 5 --seed 42
"""

from __future__ import annotations

import argparse

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from opaque.transformers.trl import DPOConfig, DPOTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DP DPO with the class-based DPOTrainer")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dataset", default="trl-lib/ultrafeedback_binarized")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--num-train-samples", type=int, default=2000)
    p.add_argument("--loss-type", nargs="+", default=["sigmoid"])
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--reference-free", action="store_true")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--microbatch-size", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--clipping-norm", type=float, default=1.0)
    p.add_argument("--noise-multiplier", type=float, default=0.8)
    p.add_argument("--target-epsilon", type=float, default=None)
    p.add_argument("--log-steps", type=int, default=5)
    p.add_argument("--output-dir", default="trainer_output/dpo")
    p.add_argument("--lora-modules", nargs="+", default=["q_proj", "v_proj"])
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_modules,
            lora_dropout=0.0,
        ),
    )
    model.print_trainable_parameters()

    # A frozen reference policy (skipped when --reference-free). With a LoRA
    # policy the base model can also serve as the reference via the PEFT
    # null-ref path (pass ref_model=None and let DPOTrainer disable the adapter).
    ref_model = (
        None
        if args.reference_free
        else AutoModelForCausalLM.from_pretrained(args.model)
    )

    raw = load_dataset(args.dataset, split=args.dataset_split, streaming=True)
    rows = [row for _, row in zip(range(args.num_train_samples), raw)]
    train_dataset = Dataset.from_list(rows)

    dpo_args = DPOConfig(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        loss_type=list(args.loss_type),
        beta=args.beta,
        reference_free=args.reference_free,
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        microbatch_size=args.microbatch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.log_steps,
        save_strategy="no",
        seed=args.seed,
        use_cpu=not torch.cuda.is_available(),
        bf16=torch.cuda.is_available(),
        # DP knobs
        clipping_norm=args.clipping_norm,
        privacy_noise_multiplier=args.noise_multiplier,
        privacy_target_epsilon=args.target_epsilon,
        optim="adamw",
        optim_args="noise_bias_correction=True",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    out = trainer.train()
    print("Training complete:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
