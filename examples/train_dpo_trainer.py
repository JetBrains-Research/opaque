"""DP Direct Preference Optimization via the class-based ``DPOTrainer``.

Trainer-based sibling of the functional ``examples/train_dpo.py``. The
functional script wires ``opaque.alignment`` DPO primitives into a hand-rolled
DP-SGD loop; this one hands the same primitives to
:class:`opaque.transformers.trl.DPOTrainer`, which precomputes the reference
log-probs and orchestrates the per-example DP path on top of ``DPTrainer``.

The reference need is derived from ``loss_type`` (there is no ``reference_free``
flag): the reference-free heads ``simpo`` / ``cpo`` / ``orpo`` (and ``sft``)
skip the reference precompute, every other head requires one. A reference is
resolved from ``--ref-model`` (a path/repo id), from the PEFT base model when
``--peft`` is set, or auto-loaded from the policy's own path.

This example exposes the DPO features that otherwise have no example coverage:
TR-DPO reference sync (``--sync-ref-model``), MPO multi-loss blends
(``--loss-type a b ... --loss-weights w1 w2 ...``), WPO weighting
(``--use-weighting``), LD-DPO (``--ld-alpha``), f-divergence regularisers
(``--f-divergence-type``), the reference-free heads, and the completion-metric
telemetry gate (``--no-log-completion-metrics``).

Examples::

    # Vanilla DPO (LoRA policy, base model as the reference)
    uv run python examples/train_dpo_trainer.py \\
      --model Qwen/Qwen2.5-0.5B-Instruct \\
      --dataset trl-lib/ultrafeedback_binarized \\
      --beta 0.1 --max-length 512 --batch-size 8 --microbatch-size 2 \\
      --max-steps 50 --learning-rate 1e-4 --peft \\
      --clipping-norm 1.0 --noise-multiplier 0.8 --log-steps 5 --seed 42

    # Reference-free SimPO (no reference precompute)
    uv run python examples/train_dpo_trainer.py --loss-type simpo --simpo-gamma 0.5

    # MPO blend: weighted sigmoid + sft regulariser
    uv run python examples/train_dpo_trainer.py \\
      --loss-type sigmoid sft --loss-weights 1.0 0.5

    # TR-DPO (EMA reference sync; full fine-tuning, not PEFT)
    uv run python examples/train_dpo_trainer.py \\
      --sync-ref-model --ref-model-mixup-alpha 0.6 --ref-model-sync-steps 64
"""

from __future__ import annotations

import argparse
import os

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig


def _configure_reporting(no_wandb: bool) -> list[str]:
    """Set W&B env defaults and return TrainingArguments.report_to.

    Matches the pattern in ``train_causal_lm_trainer.py`` so Cadence presets
    that plumb ``WANDB_NAME`` / ``WANDB_PROJECT`` / ``WANDB_ENTITY`` /
    ``WANDB_TAGS`` actually surface on the W&B dashboard. Without this, the
    HF Trainer's wandb integration only auto-init's when ``WANDB_PROJECT`` is
    set in the environment AND ``report_to`` includes ``"wandb"``; the
    DPOConfig default is ``[]``, so runs were stdout-only.
    """
    if no_wandb:
        return []
    if not os.environ.get("WANDB_MODE"):
        os.environ["WANDB_MODE"] = (
            "online" if os.environ.get("WANDB_API_KEY") else "offline"
        )
    return ["wandb"]
from transformers import AutoModelForCausalLM, AutoTokenizer

from opaque.transformers.trl import DPOConfig, DPOTrainer

# Reference-free heads (mirror ``opaque.transformers.trl._dpo_config``): a run is
# reference-free only when *every* head is in this set, so the trainer skips the
# reference precompute and ``--ref-model`` is not required.
_REFERENCE_FREE_HEADS = frozenset({"sft", "simpo", "cpo", "orpo"})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DP DPO with the class-based DPOTrainer")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument(
        "--ref-model",
        default=None,
        help="Reference policy path/repo id. Defaults to the policy's own path "
        "(auto-loaded) for reference-using heads; ignored for reference-free "
        "heads and when --peft is set (the PEFT base serves as the reference).",
    )
    p.add_argument("--dataset", default="trl-lib/ultrafeedback_binarized")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--num-train-samples", type=int, default=2000)
    # --- Loss --------------------------------------------------------------
    p.add_argument(
        "--loss-type",
        nargs="+",
        default=["sigmoid"],
        help="One or more TRL-canonical heads. Multiple ⇒ MPO blend "
        "(pair with --loss-weights). Reference-free: simpo/cpo/orpo/sft.",
    )
    p.add_argument(
        "--loss-weights",
        nargs="+",
        type=float,
        default=None,
        help="Per-head weights for an MPO blend; must match --loss-type length.",
    )
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument(
        "--f-divergence-type",
        default="reverse_kl",
        choices=["reverse_kl", "forward_kl", "js_divergence", "alpha_divergence"],
        help="f-divergence regulariser on the log-ratios.",
    )
    p.add_argument(
        "--f-alpha-divergence-coef",
        type=float,
        default=0.5,
        help="alpha coefficient for --f-divergence-type alpha_divergence.",
    )
    p.add_argument(
        "--ld-alpha",
        type=float,
        default=None,
        help="LD-DPO verbose-token weight in [0, 1]; None disables LD-DPO.",
    )
    p.add_argument(
        "--use-weighting",
        action="store_true",
        help="WPO: reweight each pair by the policy's average completion prob.",
    )
    p.add_argument("--simpo-gamma", type=float, default=0.5)
    p.add_argument("--cpo-alpha", type=float, default=1.0)
    p.add_argument("--orpo-lambda", type=float, default=1.0)
    # --- TR-DPO (reference sync) ------------------------------------------
    p.add_argument(
        "--sync-ref-model",
        action="store_true",
        help="TR-DPO: EMA-sync the reference toward the policy (full FT, not PEFT).",
    )
    p.add_argument("--ref-model-mixup-alpha", type=float, default=0.6)
    p.add_argument("--ref-model-sync-steps", type=int, default=512)
    # --- Telemetry ---------------------------------------------------------
    p.add_argument(
        "--no-log-completion-metrics",
        dest="log_completion_metrics",
        action="store_false",
        help="Skip the logits-consuming completion telemetry "
        "(entropy / mean_token_accuracy / logits/*).",
    )
    p.set_defaults(log_completion_metrics=True)
    # --- Training ----------------------------------------------------------
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--microbatch-size", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--clipping-norm", type=float, default=1.0)
    p.add_argument(
        "--noise-multiplier",
        type=float,
        default=None,
        help=(
            "Fixed noise multiplier (skips calibration). Set to 0 for a "
            "non-private baseline: the chosen mechanism/sampler are kept, no "
            "noise is added, and the accountant reports epsilon=inf. Leave "
            "unset (the default) to let ``--target-epsilon`` drive calibration."
        ),
    )
    p.add_argument(
        "--target-epsilon",
        type=float,
        default=3.0,
        help=(
            "Target ε for noise calibration. Active when ``--noise-multiplier`` "
            "is left unset; the accountant solves for the σ that achieves this "
            "ε at the configured ``(max_steps, batch_size, num_train_samples)``."
        ),
    )
    p.add_argument("--log-steps", type=int, default=5)
    p.add_argument("--output-dir", default="trainer_output/dpo")
    # --- PEFT --------------------------------------------------------------
    p.add_argument(
        "--peft",
        action="store_true",
        help="Train a LoRA adapter (the frozen base serves as the reference via "
        "the PEFT null-ref path; incompatible with --sync-ref-model).",
    )
    p.add_argument("--lora-modules", nargs="+", default=["q_proj", "v_proj"])
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    # --- Eval --------------------------------------------------------------
    p.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="If set, evaluate every N steps on a held-out slice (uses a "
        "disjoint slice of the same dataset, after the --num-train-samples).",
    )
    p.add_argument(
        "--num-eval-samples",
        type=int,
        default=200,
        help="Held-out preference-pair count for eval; ignored if --eval-steps is unset.",
    )
    p.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=None,
        help="Eval batch size; defaults to --batch-size.",
    )
    # --- Optim -------------------------------------------------------------
    p.add_argument(
        "--noise-bias-correction",
        action="store_true",
        help="Enable DP-noise-aware Adam bias correction in the opaque AdamW "
        "variant. Only meaningful at noise_multiplier > 0; default off so the "
        "noise=0 path matches stock PyTorch AdamW semantics.",
    )
    # --- W&B ---------------------------------------------------------------
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging; defaults to enabled when WANDB_PROJECT / "
        "WANDB_API_KEY env vars are set (the Cadence presets plumb these).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    loss_type = list(args.loss_type)
    reference_free = all(lt in _REFERENCE_FREE_HEADS for lt in loss_type)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)

    # PEFT policy: pass the LoRA config to the trainer (it calls get_peft_model),
    # so the frozen base can serve as the reference via the null-ref path.
    peft_config = (
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_modules,
            lora_dropout=0.0,
        )
        if args.peft
        else None
    )

    # Reference policy. Reference-free heads need none. With a PEFT policy the
    # frozen base serves as the reference (ref_model=None + null-ref path). For
    # full fine-tuning, --ref-model (a path/repo id) is loaded; DPOConfig threads
    # model_init_kwargs into this load, and ref_model may itself be a string.
    if reference_free or args.peft:
        ref_model = None
    elif args.ref_model is not None:
        ref_model = args.ref_model  # DPOTrainer accepts a string and loads it
    else:
        # Auto-load a frozen copy from the policy's own path.
        ref_model = AutoModelForCausalLM.from_pretrained(args.model)

    raw = load_dataset(args.dataset, split=args.dataset_split, streaming=True)
    eval_count = args.num_eval_samples if args.eval_steps is not None else 0
    take_total = args.num_train_samples + eval_count
    all_rows = [row for _, row in zip(range(take_total), raw)]
    train_dataset = Dataset.from_list(all_rows[: args.num_train_samples])
    eval_dataset = (
        Dataset.from_list(all_rows[args.num_train_samples : take_total])
        if eval_count
        else None
    )

    report_to = _configure_reporting(args.no_wandb)
    run_name = os.environ.get("WANDB_NAME") or os.environ.get("RUN_NAME")

    optim_args = (
        "noise_bias_correction=True" if args.noise_bias_correction else None
    )
    eval_kwargs: dict = {}
    if args.eval_steps is not None:
        eval_kwargs = {
            "eval_strategy": "steps",
            "eval_steps": args.eval_steps,
            "per_device_eval_batch_size": (
                args.per_device_eval_batch_size or args.batch_size
            ),
        }

    dpo_args = DPOConfig(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        # --- Loss ---
        loss_type=loss_type,
        loss_weights=args.loss_weights,
        beta=args.beta,
        label_smoothing=args.label_smoothing,
        f_divergence_type=args.f_divergence_type,
        f_alpha_divergence_coef=args.f_alpha_divergence_coef,
        ld_alpha=args.ld_alpha,
        use_weighting=args.use_weighting,
        simpo_gamma=args.simpo_gamma,
        cpo_alpha=args.cpo_alpha,
        orpo_lambda=args.orpo_lambda,
        # --- TR-DPO (reference sync) ---
        sync_ref_model=args.sync_ref_model,
        ref_model_mixup_alpha=args.ref_model_mixup_alpha,
        ref_model_sync_steps=args.ref_model_sync_steps,
        # --- Telemetry ---
        log_completion_metrics=args.log_completion_metrics,
        # --- Training ---
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
        # W&B
        report_to=report_to,
        run_name=run_name,
        # DP knobs
        clipping_norm=args.clipping_norm,
        privacy_noise_multiplier=args.noise_multiplier,
        privacy_target_epsilon=args.target_epsilon,
        optim="adamw",
        optim_args=optim_args,
        **eval_kwargs,
    )

    trainer = DPOTrainer(
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
