"""DP-FTRL training with matrix factorization noise for causal language models.

This script implements DP-FTRL (Follow-The-Regularized-Leader) with correlated
noise from matrix factorization mechanisms. Unlike standard DP-SGD, the noise
is correlated across steps, yielding better privacy/utility tradeoffs when
combined with the correct optimizer.

KEY DIFFERENCES FROM DP-SGD (train_causal_lm.py):

  1. Optimizer: SGD with Polyak momentum ONLY (Adam/AdaGrad are nonlinear
     operators on the gradient stream and destroy the noise correlation
     structure that MF depends on for utility gains).

  2. Clipping: Fixed norm ONLY (adaptive clipping changes sensitivity
     mid-training which invalidates the single-shot MF privacy proof).

  3. LR schedule: Linear warmup + cosine cooldown, fully predetermined
     before training starts (the optimizer must be a fixed linear map).

  4. Accounting: Single-shot from MF encoder sensitivity. No per-step
     epsilon tracking — privacy is computed once at the end.

  5. Noise: Fixed stddev, correlated across steps via C^{-1} streaming
     multiplication. Cannot change noise level mid-training.

MECHANISMS:

  band_mf   — Banded Toeplitz (Choquette-Choo et al., 2023)
               O(bands × d) memory, uses cyclic_poisson sampling.
  blt       — Buffered Linear Toeplitz (Choquette-Choo et al., 2024)
               O(buffers × d) memory, near-optimal, handles multi-epoch.
  identity  — DP-SGD baseline via MF API (C^{-1} = I, independent noise).
               Same training loop for fair comparison.

USAGE:

  # Quick smoke test (~2 minutes, GPT-2 on ag_news)
  python examples/train_dp_ftrl.py --preset smoke

  # BLT on Mellum (default mechanism, near-optimal correlated noise)
  python examples/train_dp_ftrl.py --preset mellum-kstack

  # BandMF with b=64 bands on Mellum
  python examples/train_dp_ftrl.py --preset mellum-kstack --mechanism band_mf --bands 64

  # DP-SGD baseline for fair comparison (same loop, independent noise)
  python examples/train_dp_ftrl.py --preset mellum-kstack --mechanism identity

REFERENCES:

  - BandMF: https://arxiv.org/abs/2306.08153
  - BLT: https://arxiv.org/abs/2404.16706
  - DP-FTRL: https://arxiv.org/abs/2103.00039
"""

import argparse
import contextlib
import math
import os
import sys
import time

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

import torchopt

import opaque.accounting as acc
from opaque.accounting import calibration as cal
from opaque.clipping import clipped_grad
from opaque.noise import band_mf_noise, blt_mf_noise, identity_mf_noise
from opaque.profiling import StepTimer, TrainingProfiler, print_memory, reset_peak_memory
from opaque.random import key, fold_in
from opaque.sampling import CyclicPoissonSampler, PoissonSampler, poisson_collate
from opaque.utils import make_functional

try:
    import wandb
except ImportError:
    wandb = None


# ---------------------------------------------------------------------------
# Shared utilities (same as train_causal_lm.py)
# ---------------------------------------------------------------------------


def _select_device() -> tuple[torch.device, str]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        return device, torch.cuda.get_device_name(0)
    if torch.backends.mps.is_available():
        return torch.device("mps"), "Apple Silicon"
    return torch.device("cpu"), "CPU"


def _is_dtype_supported(device: torch.device, dtype: torch.dtype) -> bool:
    try:
        torch.empty((1,), device=device, dtype=dtype)
        return True
    except (RuntimeError, TypeError):
        return False


def _resolve_model_dtype(
    requested_name: str, device: torch.device,
) -> tuple[str, torch.dtype, str | None]:
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    requested_dtype = dtype_map[requested_name]
    if _is_dtype_supported(device, requested_dtype):
        return requested_name, requested_dtype, None
    fallback_order = {
        "cuda": ["float16", "bfloat16", "float32"],
        "mps": ["float16", "float32"],
        "cpu": ["float32", "bfloat16", "float16"],
    }
    for fallback_name in fallback_order.get(device.type, ["float32"]):
        fallback_dtype = dtype_map[fallback_name]
        if _is_dtype_supported(device, fallback_dtype):
            reason = f"Requested dtype '{requested_name}' not supported on {device.type}; using '{fallback_name}'."
            return fallback_name, fallback_dtype, reason
    raise RuntimeError(f"No supported dtype for device={device.type}. Requested '{requested_name}'.")


def _load_streaming_subset(
    dataset_name: str,
    dataset_subset: str | None,
    dataset_split: str,
    dataset_text_field: str,
    total_needed: int,
) -> Dataset:
    print("  Streaming source dataset and materializing required subset...")
    stream_ds = load_dataset(
        dataset_name, name=dataset_subset, split=dataset_split, streaming=True,
    )
    rows = list(stream_ds.take(total_needed))
    if rows and dataset_text_field not in rows[0]:
        raise ValueError(
            f"Text field '{dataset_text_field}' not found. Available: {list(rows[0].keys())}"
        )
    if len(rows) < total_needed:
        raise ValueError(f"Stream ended after {len(rows)} examples, need {total_needed}.")
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def make_lr_schedule(
    base_lr: float,
    total_steps: int,
    warmup_frac: float = 0.15,
    cooldown_frac: float = 0.25,
    cooldown_end_frac: float = 0.05,
) -> torch.Tensor:
    """Create a predetermined LR schedule: linear warmup → constant → cosine cooldown.

    Args:
        base_lr: Peak learning rate.
        total_steps: Total number of training steps.
        warmup_frac: Fraction of steps for linear warmup (0→base_lr).
        cooldown_frac: Fraction of steps for cosine cooldown.
        cooldown_end_frac: LR at end of cooldown as fraction of base_lr.

    Returns:
        Tensor of shape [total_steps] with per-step learning rates.
    """
    warmup_steps = int(total_steps * warmup_frac)
    cooldown_steps = int(total_steps * cooldown_frac)
    body_steps = total_steps - warmup_steps - cooldown_steps

    if body_steps < 0:
        raise ValueError(
            f"warmup_frac ({warmup_frac}) + cooldown_frac ({cooldown_frac}) > 1.0"
        )

    schedule = torch.ones(total_steps, dtype=torch.float64)

    # Linear warmup: 0 → 1
    if warmup_steps > 0:
        schedule[:warmup_steps] = torch.linspace(0.0, 1.0, warmup_steps, dtype=torch.float64)

    # Cosine cooldown: 1 → cooldown_end_frac
    if cooldown_steps > 0:
        t = torch.linspace(0.0, math.pi, cooldown_steps, dtype=torch.float64)
        cd_min = cooldown_end_frac
        schedule[warmup_steps + body_steps:] = cd_min + (1.0 - cd_min) * 0.5 * (1.0 + torch.cos(t))

    return (schedule * base_lr).to(torch.float32)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="DP-FTRL training with matrix factorization noise"
    )

    parser.add_argument(
        "--preset", type=str, choices=["custom", "smoke", "mellum-kstack"],
        default="smoke",
        help="Preset configuration.",
    )

    # Model
    model_g = parser.add_argument_group("model")
    model_g.add_argument("--model-name", type=str, default="gpt2")
    model_g.add_argument("--attention", type=str, choices=["eager", "sdpa"], default="sdpa")
    model_g.add_argument("--sdpa-backend", type=str, choices=["flash", "efficient", "cudnn", "math"], default=None)
    model_g.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])

    # Data
    data_g = parser.add_argument_group("data")
    data_g.add_argument("--dataset", type=str, default="ag_news")
    data_g.add_argument("--dataset-subset", dest="dataset_subset", type=str, default=None)
    data_g.add_argument("--dataset-split", type=str, default="train")
    data_g.add_argument("--dataset-text-field", type=str, default="text")
    data_g.add_argument("--num-train-samples", type=int, default=5000)
    data_g.add_argument("--num-eval-samples", type=int, default=100)
    data_g.add_argument("--max-seq-len", type=int, default=512)

    # Training
    train_g = parser.add_argument_group("training")
    train_g.add_argument("--batch-size", type=int, default=16)
    train_g.add_argument("--eval-batch-size", type=int, default=None)
    train_g.add_argument("--num-epochs", type=int, default=3)
    train_g.add_argument("--learning-rate", type=float, default=5e-4)
    train_g.add_argument("--momentum", type=float, default=0.95, help="Polyak momentum (default: 0.95 per BandMF paper)")
    train_g.add_argument("--warmup-frac", type=float, default=0.15, help="LR warmup fraction (default: 0.15)")
    train_g.add_argument("--cooldown-frac", type=float, default=0.25, help="LR cooldown fraction (default: 0.25)")
    train_g.add_argument("--cooldown-end-frac", type=float, default=0.05, help="LR at end of cooldown as fraction of peak (default: 0.05)")
    train_g.add_argument("--log-steps", type=int, default=1)
    train_g.add_argument("--eval-steps", type=int, default=10)
    train_g.add_argument("--max-steps", type=int, default=None)
    train_g.add_argument("--seed", type=int, default=42)
    train_g.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    train_g.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=False)

    # LoRA
    lora_g = parser.add_argument_group("lora")
    lora_g.add_argument("--lora-r", type=int, default=4)
    lora_g.add_argument("--lora-alpha", type=int, default=8)
    lora_g.add_argument("--lora-modules", type=str, nargs="+", default=["c_attn", "c_proj"])

    # DP / MF mechanism
    dp_g = parser.add_argument_group("dp", "DP-FTRL mechanism and clipping")
    dp_g.add_argument(
        "--mechanism", type=str, default="band_mf",
        choices=["band_mf", "blt", "identity"],
        help="MF mechanism: band_mf (banded Toeplitz), blt (buffered linear Toeplitz), identity (DP-SGD baseline).",
    )
    dp_g.add_argument("--clipping-norm", type=float, default=0.9, help="Fixed clipping norm")
    dp_g.add_argument("--microbatch-size", type=int, default=None)
    dp_g.add_argument(
        "--bands", type=int, default=8,
        help="Band count for band_mf mechanism and cyclic_poisson sampling.",
    )
    dp_g.add_argument(
        "--max-buffers", type=int, default=10,
        help="Maximum BLT buffers to try (higher = better noise, slower init).",
    )

    # Privacy
    priv_g = parser.add_argument_group("privacy")
    priv_g.add_argument("--target-epsilon", type=float, default=3.0)
    priv_g.add_argument("--target-delta", type=float, default=None)
    priv_g.add_argument("--noise-multiplier", type=float, default=None, help="Fixed noise multiplier (skip calibration)")
    priv_g.add_argument("--calibration-min", type=float, default=0.1)
    priv_g.add_argument("--calibration-max", type=float, default=20.0)
    priv_g.add_argument("--calibration-tolerance", type=float, default=1e-3)

    # W&B
    track_g = parser.add_argument_group("tracking")
    track_g.add_argument("--no-wandb", action="store_true")
    track_g.add_argument("--wandb-project", type=str, default=os.environ.get("WANDB_PROJECT", "opaque"))
    track_g.add_argument("--wandb-run-name", type=str, default=os.environ.get("WANDB_NAME") or os.environ.get("RUN_NAME"))
    track_g.add_argument("--wandb-entity", type=str, default=os.environ.get("WANDB_ENTITY"))

    args = parser.parse_args()

    # Track which options were explicitly provided on CLI
    provided_dests = set()
    argv_tokens = sys.argv[1:]
    for action in parser._actions:
        if not action.option_strings:
            continue
        for opt in action.option_strings:
            if any(token == opt or token.startswith(f"{opt}=") for token in argv_tokens):
                provided_dests.add(action.dest)
                break

    def _set(name, value):
        if name not in provided_dests:
            setattr(args, name, value)

    if args.preset == "smoke":
        _set("model_name", "gpt2")
        _set("dataset", "ag_news")
        _set("dataset_text_field", "text")
        _set("num_train_samples", 1000)
        _set("num_eval_samples", 100)
        _set("num_epochs", 3)
        _set("batch_size", 32)
        _set("log_steps", 10)
        _set("eval_steps", 10)
        _set("target_epsilon", 3.0)
        _set("learning_rate", 5e-4)
        _set("lora_r", 4)
        _set("lora_alpha", 8)
        _set("max_seq_len", 512)
        _set("lora_modules", ["c_attn", "c_proj"])
        _set("dtype", "bfloat16")
    elif args.preset == "mellum-kstack":
        _set("model_name", "JetBrains/Mellum-4b-base")
        _set("dataset", "JetBrains/KStack")
        _set("dataset_text_field", "content")
        _set("num_train_samples", 500000)
        _set("num_eval_samples", 1000)
        _set("num_epochs", 8)
        _set("batch_size", 256)
        _set("log_steps", 10)
        _set("eval_steps", 25)
        _set("target_epsilon", 3.0)
        _set("learning_rate", 2e-3)
        _set("lora_r", 16)
        _set("lora_alpha", 32)
        _set("max_seq_len", 1024)
        _set("lora_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ])
        _set("dtype", "bfloat16")
        _set("microbatch_size", 16)
        _set("bands", 64)
        _set("mechanism", "blt")
        _set("warmup_frac", 0.05)
        _set("cooldown_frac", 0.30)
        _set("cooldown_end_frac", 0.01)

    if args.microbatch_size == 0:
        args.microbatch_size = None
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    print("=" * 80)
    print("DP-FTRL Training (Matrix Factorization Noise)")
    print("=" * 80)

    # --- W&B ---
    use_wandb = wandb is not None and (not args.no_wandb)
    if use_wandb:
        if args.wandb_run_name is None:
            model_short = args.model_name.split("/")[-1]
            args.wandb_run_name = (
                f"ftrl-{args.mechanism}_{model_short}_eps{args.target_epsilon}"
                f"_lr{args.learning_rate}_m{args.momentum}"
            )
        if not os.environ.get("WANDB_MODE"):
            os.environ["WANDB_MODE"] = "online" if os.environ.get("WANDB_API_KEY") else "offline"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name, config=vars(args))
        print(f"W&B initialized (mode: {os.environ.get('WANDB_MODE', 'online')})")

    # --- Device ---
    device, device_name = _select_device()
    print(f"\nDevice: {device} ({device_name})")

    torch.manual_seed(args.seed)

    # --- Attention ---
    use_eager = args.attention == "eager" or device.type == "mps"
    if not use_eager and args.sdpa_backend is not None:
        backends = {
            "flash": torch.backends.cuda.enable_flash_sdp,
            "efficient": torch.backends.cuda.enable_mem_efficient_sdp,
            "cudnn": torch.backends.cuda.enable_cudnn_sdp,
            "math": torch.backends.cuda.enable_math_sdp,
        }
        for name, setter in backends.items():
            setter(name == args.sdpa_backend)

    # --- Model ---
    print(f"\nLoading model: {args.model_name}...")
    config = AutoConfig.from_pretrained(args.model_name)
    for attr in ["attn_pdrop", "resid_pdrop", "embd_pdrop", "attention_dropout",
                 "hidden_dropout", "dropout", "attn_dropout", "ffn_dropout"]:
        if hasattr(config, attr):
            setattr(config, attr, 0.0)

    dtype_name, torch_dtype, dtype_warning = _resolve_model_dtype(args.dtype, device)
    args.dtype = dtype_name
    print(f"  Dtype: {dtype_name}")
    if dtype_warning:
        print(f"  Warning: {dtype_warning}")

    model_kwargs = {"config": config, "dtype": torch_dtype, "trust_remote_code": True}
    if use_eager:
        model_kwargs["attn_implementation"] = "eager"

    profiler = TrainingProfiler(device)

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    except TypeError as exc:
        if "unexpected keyword argument 'dtype'" not in str(exc):
            raise
        model_kwargs.pop("dtype")
        model_kwargs["torch_dtype"] = torch_dtype
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model = model.to(device)
    profiler, _ = profiler.mark("model_loaded")
    print_memory(device, "After model load")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- LoRA ---
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=args.lora_modules,
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    profiler, _ = profiler.mark("lora_applied")
    print_memory(device, "After LoRA")

    # --- Data ---
    print(f"\nLoading dataset: {args.dataset}...")
    total_needed = args.num_train_samples + args.num_eval_samples
    dataset = _load_streaming_subset(
        args.dataset, args.dataset_subset, args.dataset_split,
        args.dataset_text_field, total_needed,
    )

    eval_dataset = dataset.take(args.num_eval_samples)
    train_dataset = dataset.skip(args.num_eval_samples).take(args.num_train_samples)

    def tokenize_function(examples):
        return tokenizer(examples[args.dataset_text_field], truncation=True, max_length=args.max_seq_len)

    eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names, desc="Tokenizing eval")
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names, desc="Tokenizing train")
    print(f"Prepared: {len(train_dataset)} train, {len(eval_dataset)} eval")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    @poisson_collate
    def collate(examples):
        batch = data_collator(examples)
        return (batch["input_ids"].to(device),)

    global_train_size = len(train_dataset)

    # BLT requires fixed iteration order so consecutive participations
    # by the same example are separated by exactly steps_per_epoch steps.
    if args.mechanism == "blt":
        train_dataset = train_dataset.shuffle(seed=args.seed)

    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate, drop_last=False)

    # --- Sampling ---
    sample_rate = args.batch_size / global_train_size
    expected_steps_per_epoch = global_train_size // args.batch_size

    if args.mechanism == "band_mf":
        sampling_prob = args.batch_size * args.bands / global_train_size
        if sampling_prob > 1.0:
            raise ValueError(
                f"cyclic_poisson sampling_prob = {sampling_prob:.4f} > 1.0. "
                f"Reduce --bands ({args.bands}) or --batch-size ({args.batch_size})."
            )
    elif args.mechanism == "blt":
        pass
    elif args.mechanism == "identity":
        pass

    print("\nSampling:")
    print(f"  Mechanism: {args.mechanism}")
    if args.mechanism == "band_mf":
        print(f"  Sampler: cyclic_poisson (cycle={args.bands}, q={sampling_prob:.6f})")
    elif args.mechanism == "blt":
        print("  Sampler: epoch-based (fixed order, drop_last=True)")
        print(f"  min_sep: {expected_steps_per_epoch} (= steps/epoch)")
        print(f"  max_participations: {args.num_epochs}")
    else:
        print(f"  Sampler: poisson (q={sample_rate:.6f})")
    print(f"  Expected batch size: {args.batch_size}")
    print(f"  Expected steps/epoch: {expected_steps_per_epoch}")

    # --- Gradient checkpointing ---
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("\nGradient checkpointing: enabled")

    offload_ctx = (
        torch.autograd.graph.save_on_cpu(pin_memory=True)
        if args.cpu_offload else contextlib.nullcontext()
    )

    # --- Functional conversion ---
    print("\nConverting to functional form...")
    t0 = time.time()
    fmodel, trainable_params, frozen_params = make_functional(
        model, disable_autograd_tracking=True, partition_trainable=True,
    )
    param_names = list(trainable_params.keys())
    print(f"Trainable parameters: {len(param_names)} (took {time.time() - t0:.1f}s)")
    profiler, _ = profiler.mark("functional_conversion")
    print_memory(device, "After functional conversion")

    def merged_params(trainable):
        return {**frozen_params, **trainable}

    def per_example_loss_fn(trainable, input_ids):
        output = fmodel(merged_params(trainable), input_ids, labels=input_ids)
        return output.loss

    def eval_loss(trainable):
        with torch.no_grad():
            total_loss, total_count = 0.0, 0
            for (input_ids,) in eval_loader:
                loss = per_example_loss_fn(trainable, input_ids)
                total_loss += loss.item() * len(input_ids)
                total_count += len(input_ids)
            return total_loss / total_count

    grad_fn, clip_state = clipped_grad(
        per_example_loss_fn,
        argnums=0,
        batch_argnums=(1,),
        clipping_norm=args.clipping_norm,
        normalize_by=args.batch_size,
        microbatch_size=args.microbatch_size,
        return_aux=True,
    )

    # --- Total steps & LR schedule ---
    total_steps = args.num_epochs * expected_steps_per_epoch
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    lr_schedule = make_lr_schedule(
        args.learning_rate, total_steps,
        warmup_frac=args.warmup_frac,
        cooldown_frac=args.cooldown_frac,
        cooldown_end_frac=args.cooldown_end_frac,
    )

    print(f"\nLR schedule: warmup {args.warmup_frac:.0%} → constant → cooldown {args.cooldown_frac:.0%}")
    print(f"  Peak LR: {args.learning_rate}")
    print(f"  Cooldown end: {args.cooldown_end_frac:.0%} of peak")
    print(f"  Total steps: {total_steps}")

    # --- Privacy calibration (single-shot) ---
    if args.target_delta is None:
        args.target_delta = 1.0 / (global_train_size ** 1.1)

    # Build the accounting mechanism
    if args.mechanism == "band_mf":
        def acct_mechanism(nm):
            return acc.cyclic_poisson(
                acc.band_mf(nm, n_steps=total_steps, bands=args.bands,
                            momentum=args.momentum),
                sample_rate=sampling_prob,
            )
    elif args.mechanism == "blt":
        def acct_mechanism(nm):
            return acc.blt_mf(
                nm, n_steps=total_steps,
                min_sep=expected_steps_per_epoch,
                max_participations=args.num_epochs,
                max_buffers=args.max_buffers,
                momentum=args.momentum,
            )
    elif args.mechanism == "identity":
        def acct_mechanism(nm):
            return acc.poisson(acc.gaussian(nm), sample_rate=sample_rate) * total_steps
    else:
        raise ValueError(f"Unknown mechanism: {args.mechanism}")

    if args.noise_multiplier is not None:
        noise_multiplier = args.noise_multiplier
        print(f"\nFixed noise multiplier: {noise_multiplier:.4f}")
    else:
        print(f"\nCalibrating noise (mechanism={args.mechanism})...")
        print(f"  Target: ε={args.target_epsilon}, δ={args.target_delta:.2e}")
        t0 = time.time()
        cal_min = args.calibration_min
        cal_max = args.calibration_max
        calibration = cal.calibrate(
            cal.epsilon_budget(args.target_epsilon, delta=args.target_delta),
            acct_mechanism,
            param_min=cal_min,
            param_max=cal_max,
            tolerance=args.calibration_tolerance,
        )
        noise_multiplier = calibration.param
        print(f"  Calibrated in {time.time() - t0:.1f}s: σ={noise_multiplier:.4f} (ε≈{calibration.achieved:.3f})")

    if use_wandb:
        wandb.config.update({
            "noise_multiplier": noise_multiplier,
            "target_delta": args.target_delta,
            "total_steps": total_steps,
        }, allow_val_change=True)

    # --- Create MF noise function (fixed stddev) ---
    # sensitivity = clipping_norm / normalize_by (accounts for batch averaging)
    noise_stddev = noise_multiplier * clip_state.sensitivity

    print(f"\nCreating MF noise (optimizing for momentum-SGD workload, β={args.momentum})...")
    t0 = time.time()
    if args.mechanism == "band_mf":
        noise_fn, noise_state = band_mf_noise(
            trainable_params, total_steps,
            stddev=noise_stddev, key=key(args.seed), bands=args.bands,
            momentum=args.momentum,
        )
    elif args.mechanism == "blt":
        noise_fn, noise_state = blt_mf_noise(
            trainable_params, total_steps,
            stddev=noise_stddev, key=key(args.seed),
            min_sep=expected_steps_per_epoch,
            max_participations=args.num_epochs,
            max_buffers=args.max_buffers,
            momentum=args.momentum,
        )
    elif args.mechanism == "identity":
        noise_fn, noise_state = identity_mf_noise(
            trainable_params, stddev=noise_stddev, key=key(args.seed),
        )
    print(f"  Noise function created in {time.time() - t0:.1f}s")

    optimizer = torchopt.sgd(
        lr=lambda step: lr_schedule[min(step, len(lr_schedule) - 1)].item(),
        momentum=args.momentum,
    )
    opt_state = optimizer.init(trainable_params)

    # --- Diagnostic: compute what identity baseline σ would be ---
    identity_sigma = None
    if args.mechanism != "identity" and args.noise_multiplier is None:
        try:
            identity_cal = cal.calibrate(
                cal.epsilon_budget(args.target_epsilon, delta=args.target_delta),
                lambda nm: acc.poisson(acc.gaussian(nm), sample_rate=sample_rate) * total_steps,
                param_min=args.calibration_min,
                param_max=args.calibration_max,
                tolerance=args.calibration_tolerance,
            )
            identity_sigma = identity_cal.param
        except Exception:
            pass  # Non-critical diagnostic

    print("\nDP-FTRL setup:")
    print(f"  Mechanism: {args.mechanism}")
    print(f"  Optimizer: SGD + Polyak momentum (β={args.momentum})")
    print(f"  Workload: momentum-SGD (β={args.momentum}){' [prefix-sum]' if args.momentum == 1.0 else ''}")
    print(f"  Clipping norm: {args.clipping_norm} (fixed)")
    print(f"  Sensitivity: {clip_state.sensitivity:.6f} (= {args.clipping_norm} / {args.batch_size})")
    print(f"  Noise multiplier (σ): {noise_multiplier:.4f}")
    print(f"  Noise stddev: {noise_stddev:.6f} (= {noise_multiplier:.4f} × {clip_state.sensitivity:.6f})")
    if identity_sigma is not None:
        ratio = noise_multiplier / identity_sigma
        print(f"  Identity baseline σ: {identity_sigma:.4f} (ratio: {ratio:.2f}×)")
        print(f"  → MF needs {ratio:.2f}× more noise to hit ε={args.target_epsilon}; "
              f"correlated structure must compensate")
    print(f"  Microbatch size: {args.microbatch_size}")
    if args.mechanism == "band_mf":
        print(f"  Bands: {args.bands}")
    elif args.mechanism == "blt":
        print(f"  Max buffers: {args.max_buffers}")
        print(f"  Min separation: {expected_steps_per_epoch}")
        print(f"  Max participations: {args.num_epochs}")

    # ===================================================================
    # Training loop
    # ===================================================================
    print("\n" + "=" * 80)
    print("Starting DP-FTRL training...")
    print("=" * 80)

    losses = []
    clip_rates = []
    global_step = 0

    reset_peak_memory(device)
    profiler, _ = profiler.mark("training_start")
    print_memory(device, "Before training")

    # Step-0 eval
    initial_eval_loss = eval_loss(trainable_params)
    print(f"  → Step 0 eval: loss={initial_eval_loss:.4f}")
    if use_wandb:
        wandb.log({"eval/loss": initial_eval_loss, "train/lr": lr_schedule[0].item()}, step=0)

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 80)

        # Create epoch data loader — each mechanism needs a different sampler.
        if args.mechanism == "band_mf":
            epoch_sampler = CyclicPoissonSampler(
                train_dataset,
                sampling_prob=sampling_prob,
                cycle_length=args.bands,
                iterations=expected_steps_per_epoch,
                key=fold_in(key(args.seed), epoch),
            )
            epoch_loader = DataLoader(train_dataset, batch_sampler=epoch_sampler, collate_fn=collate)
        elif args.mechanism == "blt":
            epoch_loader = DataLoader(
                train_dataset, batch_size=args.batch_size,
                shuffle=False, collate_fn=collate, drop_last=True,
            )
        else:  # identity
            epoch_sampler = PoissonSampler(
                train_dataset,
                sample_rate=sample_rate,
                num_iterations=expected_steps_per_epoch,
                key=fold_in(key(args.seed), epoch),
            )
            epoch_loader = DataLoader(train_dataset, batch_sampler=epoch_sampler, collate_fn=collate)

        for step_idx, batch in enumerate(epoch_loader):
            if args.max_steps is not None and global_step >= args.max_steps:
                break

            (input_ids,) = batch
            batch_size = len(input_ids)

            lr_t = lr_schedule[min(global_step, len(lr_schedule) - 1)].item()

            step_timer = StepTimer(device, batch_size=batch_size)
            with step_timer:
                with offload_ctx:
                    (grads, aux), clip_state = grad_fn(
                        trainable_params, input_ids, state=clip_state,
                    )

                noisy_grads, noise_state = noise_fn(grads, noise_state)
                updates, opt_state = optimizer.update(
                    noisy_grads, opt_state, params=trainable_params,
                )
                trainable_params = torchopt.apply_updates(trainable_params, updates)

            profiler = profiler.add_step(step_timer)

            if batch_size == 0:
                global_step += 1
                continue

            # --- Step metrics ---
            avg_loss = aux.loss_values.mean().item()
            clip_rate = aux.clipping_rate
            mean_grad_norm = aux.grad_norms.mean().item()
            losses.append(avg_loss)
            clip_rates.append(clip_rate)
            global_step += 1

            # --- Logging ---
            if global_step % args.log_steps == 0:
                perf = profiler.current_metrics()

                if use_wandb:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/batch_size": batch_size,
                        "train/clipping_norm": args.clipping_norm,
                        "train/clip_rate": clip_rate,
                        "train/grad_norm_mean": mean_grad_norm,
                        "train/noise_std": noise_stddev,
                        "train/lr": lr_t,
                        "train/momentum": args.momentum,
                        "perf/step_time_sec": perf["step_time_sec"],
                        "perf/throughput_samples_per_sec": perf["throughput_samples_sec"],
                        "perf/peak_gb": perf["memory_peak_gb"],
                    }, step=global_step)

                print(
                    f"Step {global_step:4d} [E{epoch+1} S{step_idx+1:3d}/{expected_steps_per_epoch:3d}] | "
                    f"BS: {batch_size} | Loss: {avg_loss:.4f} | "
                    f"Clip: {clip_rate:.1%} | GradNorm: {mean_grad_norm:.3f} | "
                    f"LR: {lr_t:.2e} | "
                    f"Time: {perf['step_time_sec']:.2f}s | Mem: {perf['memory_peak_gb']:.1f}GB"
                )

            # --- Eval ---
            if global_step % args.eval_steps == 0:
                current_eval_loss = eval_loss(trainable_params)
                print(f"  → Eval: loss={current_eval_loss:.4f}")
                if use_wandb:
                    wandb.log({"eval/loss": current_eval_loss}, step=global_step)

        if args.max_steps is not None and global_step >= args.max_steps:
            print(f"\nReached max_steps={args.max_steps}, stopping.")
            break

    # ===================================================================
    # Final summary
    # ===================================================================
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset} ({global_train_size} train samples)")
    print(f"\nMechanism: {args.mechanism}")
    print(f"Optimizer: SGD + Polyak momentum (β={args.momentum})")
    print("\nTraining results:")
    print(f"  Total steps: {global_step}")
    if losses:
        print(f"  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
    if clip_rates:
        print("\nClipping:")
        print(f"  Fixed norm: {args.clipping_norm}")
        print(f"  Average clip rate: {sum(clip_rates) / len(clip_rates):.2%}")

    # Single-shot accounting
    final_epsilon = acct_mechanism(noise_multiplier).epsilon_at(args.target_delta)
    print("\nPrivacy (single-shot):")
    print(f"  Target: ε={args.target_epsilon:.3f}, δ={args.target_delta:.2e}")
    print(f"  Noise multiplier: {noise_multiplier:.4f}")
    print(f"  Final ε: {final_epsilon:.4f}")

    if use_wandb:
        wandb.log({"privacy/epsilon_final": final_epsilon}, step=global_step)

    profiler, _ = profiler.mark("training_complete")
    print("\n" + profiler.final_summary())
    print("\n" + profiler.checkpoint_summary())

    if use_wandb:
        wandb.finish()

    return 0


if __name__ == "__main__":
    exit(main())
