"""DP Supervised Fine-Tuning via the class-based ``SFTTrainer``.

Trainer-based sibling of the functional ``examples/train_sft.py`` (the same way
``train_causal_lm_trainer.py`` mirrors ``train_causal_lm.py``). The functional
script wires ``opaque.alignment`` primitives into a hand-rolled DP-SGD loop;
this one hands the same primitives to :class:`opaque.transformers.trl.SFTTrainer`,
which orchestrates the per-example DP path on top of ``DPTrainer``.

Example::

    uv run python examples/train_sft_trainer.py \\
      --model-name Qwen/Qwen2.5-0.5B \\
      --dataset roneneldan/TinyStories --dataset-text-field text \\
      --num-train-samples 2000 --loss-type nll \\
      --max-length 512 --batch-size 8 --num-steps 50 \\
      --learning-rate 1e-4 --clipping-norm 1.0 --noise-multiplier 0.8 \\
      --lora-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \\
      --seed 42
"""

from __future__ import annotations

import argparse

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from opaque.transformers.trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DP SFT with the class-based SFTTrainer")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--dataset", default="roneneldan/TinyStories")
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--dataset-text-field", default="text")
    p.add_argument("--num-train-samples", type=int, default=2000)
    p.add_argument("--loss-type", default="nll", choices=["nll", "dft"])
    p.add_argument("--completion-only", action="store_true")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--microbatch-size", type=int, default=None)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--clipping-norm", type=float, default=1.0)
    p.add_argument("--noise-multiplier", type=float, default=0.8)
    p.add_argument("--target-epsilon", type=float, default=None)
    p.add_argument("--log-steps", type=int, default=5)
    p.add_argument("--output-dir", default="trainer_output/sft")
    p.add_argument("--lora-modules", nargs="+", default=["q_proj", "v_proj"])
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
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

    raw = load_dataset(
        args.dataset, args.dataset_config, split=args.dataset_split, streaming=True
    )
    rows = [row for _, row in zip(range(args.num_train_samples), raw)]
    train_dataset = Dataset.from_list(rows)

    sft_args = SFTConfig(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        dataset_text_field=args.dataset_text_field,
        completion_only_loss=True if args.completion_only else None,
        loss_type=args.loss_type,
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        microbatch_size=args.microbatch_size,
        max_steps=args.num_steps,
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

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    out = trainer.train()
    print("Training complete:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
