"""DP-FTRL training with matrix factorization noise for causal language models.

This script implements DP-FTRL (Follow-The-Regularized-Leader) with correlated
noise from matrix factorization mechanisms. Unlike standard DP-SGD, the noise
is correlated across steps, yielding better privacy/utility tradeoffs when
combined with the correct optimizer.

KEY DIFFERENCES FROM DP-SGD (train_causal_lm.py):

  1. Optimizer: SGD with Polyak momentum (default), or one of the
     Opaque-built v-using optimizers (``adamw``, ``ademamix``).  For
    adaptive optimizers, pair with ``--second-moment`` to activate a
    private squared-gradient stream.  The optimizer consumes the
    privately-estimated ``g²`` stream alongside standard noised gradients.
    ``lion`` is also exposed but has no v, so ``--second-moment`` is
    rejected for it.

  2. Clipping: Fixed scalar norm or fixed per-group norms (``--per-group-clipping``).
     Adaptive clipping is not supported (sensitivity must stay constant across
     the MF run for the single-shot privacy proof).

  3. LR schedule: Constant by default (--warmup-frac 0); optional linear warmup
     → constant, fully predetermined before training (fixed linear map).

  4. Accounting: Single-shot from MF encoder sensitivity. No per-step
     epsilon tracking — privacy is computed once at the end.

  5. Noise: Fixed stddev, correlated across steps via C^{-1} streaming
     multiplication. Cannot change noise level mid-training.
    With ``--second-moment`` (Adam-family only): two independent MF noise
    streams, calibrated via joint first+second moment sensitivity.

MECHANISMS:

  band_mf   — Banded Toeplitz (Choquette-Choo et al., 2023)
               O(bands × d) memory, uses Poisson (cyclic) sampling.
  blt       — Buffered Linear Toeplitz (Choquette-Choo et al., 2024)
               O(buffers × d) memory, near-optimal, handles multi-epoch.
  lambda_cgd — DP-λCGD (Kalinin et al., 2026), bandwidth-2 correlated noise
               via PRNG replay. Single hyperparam λ, zero extra memory.
  bisr      — BISR (Kalinin et al., ICLR 2026), generalises λCGD to
               arbitrary bandwidth p. Coefficients from inverse square root.
  bsr       — BSR (Kalinin & Lampert, NeurIPS 2024), closed-form banded square
               root for the paper workload (α, β); no L-BFGS. BnB sampling.
  identity  — DP-SGD baseline via MF API (C^{-1} = I, independent noise).
               Same training loop for fair comparison.

USAGE:

  # Quick smoke test (~2 minutes, GPT-2 on ag_news)
  python examples/train_dp_ftrl.py --preset smoke

  # BLT on Mellum (default mechanism, near-optimal correlated noise)
  python examples/train_dp_ftrl.py --preset mellum-kstack

  # BandMF with b=64 bands on Mellum
  python examples/train_dp_ftrl.py --preset mellum-kstack --mechanism band_mf --bands 64

  # DP-λCGD with Balls-in-Bins sampling (bandwidth-2 correlated noise, λ=0.9)
  python examples/train_dp_ftrl.py --preset mellum-kstack --mechanism lambda_cgd --lambda_ 0.9

  # BISR with bandwidth=4, Balls-in-Bins sampling
  python examples/train_dp_ftrl.py --preset mellum-kstack --mechanism bisr --bisr-bandwidth 4

  # BSR (closed-form): workload α via --bsr-alpha (paper default 1.0); optimizer WD is separate (--weight-decay, default 0)
  python examples/train_dp_ftrl.py --preset smoke --mechanism bsr --bsr-bandwidth 8 --bsr-alpha 1.0

  # DP-SGD baseline for fair comparison (same loop, independent noise)
  python examples/train_dp_ftrl.py --preset mellum-kstack --mechanism identity

  # Non-DP baseline (no noise, no privacy accounting, same loop)
  python examples/train_dp_ftrl.py --preset mellum-kstack --mechanism none

    # Adam-family without private second moments (single-stream MF noise)
  python examples/train_dp_ftrl.py --preset smoke --optimizer adamw

    # DP-Adam with private second moments (two MF noise streams)
    python examples/train_dp_ftrl.py --preset smoke --optimizer adamw --second-moment
    python examples/train_dp_ftrl.py --preset smoke --optimizer adamw --second-moment --mechanism blt
    python examples/train_dp_ftrl.py --preset smoke --optimizer adamw --second-moment --beta1 0.9 --beta2 0.999

    # AdEMAMix with private second moments — slow EMA captures long-range gradient signal
    python examples/train_dp_ftrl.py --preset smoke --optimizer ademamix --second-moment

    # Lion under MF noise (no private second moment — lion has no second moment)
  python examples/train_dp_ftrl.py --preset smoke --optimizer lion

REFERENCES:

  - BandMF: https://arxiv.org/abs/2306.08153
  - BLT: https://arxiv.org/abs/2404.16706
  - DP-λCGD: https://arxiv.org/abs/2601.22334
  - BISR: https://arxiv.org/abs/2505.12128
  - BSR: https://arxiv.org/abs/2405.13763
  - DP-FTRL: https://arxiv.org/abs/2103.00039
    - Private second moments: https://arxiv.org/abs/2502.06597
"""

# ruff: noqa: E402

import argparse
import contextlib
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

from opaque.patches import apply_model_patches, apply_runtime_patches

apply_runtime_patches()

import torchopt

import opaque.accounting as acc
import opaque.dpftrl.accounting as ftrl_acc
from opaque.accounting import calibration as cal
from opaque.dpftrl.clipping import clipped_grad, per_group
from opaque.types import PerGroup, SecondMomentNoiseOutput
from opaque.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    bsr_strategy,
    blt_strategy,
    identity_strategy,
    lambda_cgd_strategy,
    mf_noise,
)
from opaque.profiling import (
    StepTimer,
    TrainingProfiler,
    print_memory,
    reset_peak_memory,
)
from opaque.random import key, fold_in
from opaque.functional import empty_collate
from opaque.dpftrl.sampling import (
    BallsInBinsSampler,
    BMinSepSampler,
    PoissonSampler,
    SequentialBatchSampler,
)
from opaque.functional import make_functional

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
    requested_name: str,
    device: torch.device,
) -> tuple[str, torch.dtype, str | None]:
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
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
    raise RuntimeError(
        f"No supported dtype for device={device.type}. Requested '{requested_name}'."
    )


def _load_streaming_subset(
    dataset_name: str,
    dataset_subset: str | None,
    dataset_split: str,
    dataset_text_field: str,
    total_needed: int,
) -> Dataset:
    print("  Streaming source dataset and materializing required subset...")
    stream_ds = load_dataset(
        dataset_name,
        name=dataset_subset,
        split=dataset_split,
        streaming=True,
    )
    rows = list(stream_ds.take(total_needed))
    if rows and dataset_text_field not in rows[0]:
        raise ValueError(
            f"Text field '{dataset_text_field}' not found. Available: {list(rows[0].keys())}"
        )
    if len(rows) < total_needed:
        raise ValueError(
            f"Stream ended after {len(rows)} examples, need {total_needed}."
        )
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def make_lr_schedule(
    base_lr: float,
    total_steps: int,
    warmup_frac: float = 0.0,
) -> torch.Tensor:
    """Create a predetermined LR schedule: optional linear warmup → constant.

    Args:
        base_lr: Peak learning rate.
        total_steps: Total number of training steps.
        warmup_frac: Fraction of steps for linear warmup (0→base_lr). Default 0
            (constant schedule at base_lr).

    Returns:
        Tensor of shape [total_steps] with per-step learning rates.
    """
    warmup_steps = int(total_steps * warmup_frac)

    schedule = torch.ones(total_steps, dtype=torch.float64)

    # Linear warmup: 0 → 1
    if warmup_steps > 0:
        schedule[:warmup_steps] = torch.linspace(
            0.0, 1.0, warmup_steps, dtype=torch.float64
        )

    return (schedule * base_lr).to(torch.float32)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="DP-FTRL training with matrix factorization noise"
    )

    parser.add_argument(
        "--preset",
        type=str,
        choices=["custom", "smoke", "mellum-kstack"],
        default="smoke",
        help="Preset configuration.",
    )

    # Model
    model_g = parser.add_argument_group("model")
    model_g.add_argument("--model-name", type=str, default="gpt2")
    model_g.add_argument(
        "--attention", type=str, choices=["eager", "sdpa"], default="sdpa"
    )
    model_g.add_argument(
        "--sdpa-backend",
        type=str,
        choices=["flash", "efficient", "cudnn", "math"],
        default=None,
    )
    model_g.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )

    # Data
    data_g = parser.add_argument_group("data")
    data_g.add_argument("--dataset", type=str, default="ag_news")
    data_g.add_argument(
        "--dataset-subset", dest="dataset_subset", type=str, default=None
    )
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
    train_g.add_argument(
        "--optimizer",
        type=str,
        choices=["sgd", "adamw", "ademamix", "lion"],
        default="sgd",
        help=(
            "Optimizer.  ``sgd`` is the canonical DP-FTRL baseline "
            "(sgd, Polyak momentum).  ``adamw`` and ``ademamix`` "
            "are Adam-family adaptive optimizers; pair with ``--second-moment`` to "
            "activate a private squared-gradient stream.  ``lion`` "
            "is sign-of-momentum; works under MF noise but has no v so "
            "``--second-moment`` is rejected for it."
        ),
    )
    train_g.add_argument(
        "--second-moment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Activate private second-moment noise: ``mf_noise`` produces a "
            "privately-estimated ``g²`` stream alongside noised gradients, "
            "and Opaque optimizers consume it automatically.  Joint noise "
            "uses sensitivity-proportional Mahalanobis allocation: privacy "
            "accounting is the underlying MF mechanism at the same noise "
            "multiplier — no extra cost.  Requires an Adam-family optimizer "
            "(``adamw`` or ``ademamix``) with a second moment to consume "
            "the paired stream.  Off by default."
        ),
    )
    train_g.add_argument(
        "--momentum",
        type=float,
        default=0.95,
        help="Polyak momentum for SGD (default: 0.95 per BandMF paper)",
    )
    train_g.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Optimizer weight decay: sgd L2-style coefficient, or "
        "adamw decoupled WD (default 0).",
    )
    train_g.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Adam first moment decay (default: 0.9). Ignored for --optimizer sgd.",
    )
    train_g.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="Adam second moment decay (default: 0.999). Ignored for --optimizer sgd.",
    )
    train_g.add_argument(
        "--adam-eps",
        type=float,
        default=1e-8,
        help="Adam epsilon (default: 1e-8). Ignored for --optimizer sgd.",
    )
    train_g.add_argument(
        "--warmup-frac",
        type=float,
        default=0.0,
        help="LR warmup fraction of total steps, 0→peak LR linearly (default: 0 = constant LR)",
    )
    train_g.add_argument("--log-steps", type=int, default=1)
    train_g.add_argument("--eval-steps", type=int, default=10)
    train_g.add_argument("--max-steps", type=int, default=None)
    train_g.add_argument("--seed", type=int, default=42)
    train_g.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False
    )
    train_g.add_argument(
        "--cpu-offload", action=argparse.BooleanOptionalAction, default=False
    )

    # LoRA
    lora_g = parser.add_argument_group("lora")
    lora_g.add_argument("--lora-r", type=int, default=4)
    lora_g.add_argument("--lora-alpha", type=int, default=8)
    lora_g.add_argument(
        "--lora-modules", type=str, nargs="+", default=["c_attn", "c_proj"]
    )

    # DP / MF mechanism
    dp_g = parser.add_argument_group("dp", "DP-FTRL mechanism and clipping")
    dp_g.add_argument(
        "--mechanism",
        type=str,
        default="band_mf",
        choices=["band_mf", "blt", "lambda_cgd", "bisr", "bsr", "identity", "none"],
        help="MF mechanism: band_mf, blt, lambda_cgd, bisr, bsr, identity, none (non-DP).",
    )
    dp_g.add_argument(
        "--clipping-norm", type=float, default=0.9, help="Fixed clipping norm"
    )
    dp_g.add_argument(
        "--per-group-clipping",
        type=str,
        nargs="+",
        default=None,
        metavar="PATTERN=NORM",
        help="Per-group clipping norms as PATTERN=NORM pairs (e.g. c_attn=0.9 c_proj=0.5 "
        "for --preset smoke GPT-2 LoRA, or q_proj=0.5 fallback=1.0 for Mellum presets). "
        "Each trainable param must match exactly one pattern substring. "
        "Use 'fallback=NORM' as catch-all.  Incompatible with adaptive clipping; "
        "MF ``mf_noise`` uses the same Mahalanobis allocation as DP-SGD Gaussian.",
    )
    dp_g.add_argument("--microbatch-size", type=int, default=None)
    dp_g.add_argument(
        "--bands",
        type=int,
        default=8,
        help="Band count for band_mf mechanism and BandMF subsampling.",
    )
    dp_g.add_argument(
        "--band-mf-sampling",
        type=str,
        choices=["poisson", "b_min_sep"],
        default="poisson",
        help="BandMF data subsampling: poisson (default) or b_min_sep (Dong & Ganesh 2026).",
    )
    dp_g.add_argument(
        "--mc-samples",
        type=int,
        default=100_000,
        help="Monte Carlo samples for MC-based privacy accounting (b_min_sep, BnB).",
    )
    dp_g.add_argument(
        "--max-buffers",
        type=int,
        default=10,
        help="Maximum BLT buffers to try (higher = better noise, slower init).",
    )
    dp_g.add_argument(
        "--lambda_",
        type=float,
        default=0.9,
        help="Correlation coefficient for lambda_cgd mechanism (0=DP-SGD, higher=more correlation).",
    )
    dp_g.add_argument(
        "--bisr-bandwidth",
        type=int,
        default=4,
        help="Bandwidth for BISR mechanism (>= 2). Higher = better utility, more PRNG replays.",
    )
    dp_g.add_argument(
        "--bsr-bandwidth",
        type=int,
        default=8,
        help="Bandwidth p for BSR mechanism (>= 1). Closed-form coefficients; no optimizer.",
    )
    dp_g.add_argument(
        "--bsr-alpha",
        type=float,
        default=1.0,
        help="BSR workload α in (0,1] (paper); must satisfy α>β where β comes from "
        "--momentum (SGD) or --beta1 (Adam). Ignored unless --mechanism bsr.",
    )

    # Privacy
    priv_g = parser.add_argument_group("privacy")
    priv_g.add_argument("--target-epsilon", type=float, default=3.0)
    priv_g.add_argument("--target-delta", type=float, default=None)
    priv_g.add_argument(
        "--noise-multiplier",
        type=float,
        default=None,
        help="Fixed noise multiplier (skip calibration)",
    )
    priv_g.add_argument("--calibration-min", type=float, default=0.1)
    priv_g.add_argument("--calibration-max", type=float, default=20.0)
    priv_g.add_argument("--calibration-tolerance", type=float, default=1e-3)

    # W&B
    track_g = parser.add_argument_group("tracking")
    track_g.add_argument("--no-wandb", action="store_true")
    track_g.add_argument(
        "--wandb-project", type=str, default=os.environ.get("WANDB_PROJECT", "opaque")
    )
    track_g.add_argument(
        "--wandb-run-name",
        type=str,
        default=os.environ.get("WANDB_NAME") or os.environ.get("RUN_NAME"),
    )
    track_g.add_argument(
        "--wandb-entity", type=str, default=os.environ.get("WANDB_ENTITY")
    )

    args = parser.parse_args()

    # Track which options were explicitly provided on CLI
    provided_dests = set()
    argv_tokens = sys.argv[1:]
    for action in parser._actions:
        if not action.option_strings:
            continue
        for opt in action.option_strings:
            if any(
                token == opt or token.startswith(f"{opt}=") for token in argv_tokens
            ):
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
        _set(
            "lora_modules",
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        _set("dtype", "bfloat16")
        _set("microbatch_size", 16)
        _set("bands", 64)
        _set("mechanism", "blt")
        _set("warmup_frac", 0.05)
    if args.microbatch_size == 0:
        args.microbatch_size = None
    if args.eval_batch_size is None:
        args.eval_batch_size = args.microbatch_size or args.batch_size

    if args.per_group_clipping:
        parsed: dict[str, float] = {}
        fallback_value = None
        for item in args.per_group_clipping:
            if "=" not in item:
                parser.error(
                    f"--per-group-clipping values must be PATTERN=NORM, got '{item}'"
                )
            pattern, value = item.split("=", 1)
            try:
                norm = float(value)
            except ValueError:
                parser.error(
                    f"--per-group-clipping norm must be a number, got '{value}' "
                    f"in '{item}'"
                )
            if pattern == "fallback":
                fallback_value = norm
            else:
                parsed[pattern] = norm
        args.per_group_clipping = parsed
        args.per_group_clipping_fallback = fallback_value

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
            os.environ["WANDB_MODE"] = (
                "online" if os.environ.get("WANDB_API_KEY") else "offline"
            )
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
        )
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
    for attr in [
        "attn_pdrop",
        "resid_pdrop",
        "embd_pdrop",
        "attention_dropout",
        "hidden_dropout",
        "dropout",
        "attn_dropout",
        "ffn_dropout",
    ]:
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
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    apply_model_patches(model)
    model.print_trainable_parameters()
    profiler, _ = profiler.mark("lora_applied")
    print_memory(device, "After LoRA")

    # --- Data ---
    print(f"\nLoading dataset: {args.dataset}...")
    total_needed = args.num_train_samples + args.num_eval_samples
    dataset = _load_streaming_subset(
        args.dataset,
        args.dataset_subset,
        args.dataset_split,
        args.dataset_text_field,
        total_needed,
    )

    eval_dataset = dataset.take(args.num_eval_samples)
    train_dataset = dataset.skip(args.num_eval_samples).take(args.num_train_samples)

    def tokenize_function(examples):
        return tokenizer(
            examples[args.dataset_text_field],
            truncation=True,
            max_length=args.max_seq_len,
        )

    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing eval",
    )
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
    )
    print(f"Prepared: {len(train_dataset)} train, {len(eval_dataset)} eval")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    @empty_collate
    def collate(examples):
        batch = data_collator(examples)
        return (batch["input_ids"].to(device),)

    global_train_size = len(train_dataset)

    # BLT uses fixed iteration order (sequential DataLoader, drop_last=True),
    # so shuffle once to randomize which examples land in which batch.
    # λ-CGD and BISR use BnB sampling which randomizes assignment itself.
    if args.mechanism == "blt":
        train_dataset = train_dataset.shuffle(seed=args.seed)

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        drop_last=False,
    )

    # --- Sampling ---
    sample_rate = args.batch_size / global_train_size
    expected_steps_per_epoch = global_train_size // args.batch_size

    # Create sampler (or per-epoch sampler factory).
    # Static samplers (BnB, sequential) are created once and reused.
    # Dynamic samplers (Poisson) get a fresh key each epoch.
    if args.mechanism == "band_mf":
        p0 = sample_rate  # E[batch]/|D| per iteration (same as cyclic Poisson regime)
        sampling_prob = 0.0
        p_bms = 0.0
        if args.band_mf_sampling == "poisson":
            sampling_prob = args.batch_size * args.bands / global_train_size
            if sampling_prob > 1.0:
                raise ValueError(
                    f"poisson sampling_prob = {sampling_prob:.4f} > 1.0. "
                    f"Reduce --bands ({args.bands}) or --batch-size ({args.batch_size})."
                )

            def make_epoch_sampler(epoch):
                return PoissonSampler(
                    train_dataset,
                    sample_rate=sampling_prob,
                    bands=args.bands,
                    n_steps=expected_steps_per_epoch,
                    key=fold_in(key(args.seed), epoch),
                )
        else:
            if args.bands > 1 and p0 * (args.bands - 1) >= 1.0:
                raise ValueError(
                    f"b_min_sep requires p_0 < 1/(bands-1); got p_0={p0:.6f}, bands={args.bands}."
                )
            denom = 1.0 - p0 * max(0, args.bands - 1)
            p_bms = p0 / denom if args.bands > 1 else p0
            if p_bms > 1.0:
                raise ValueError(
                    f"b_min_sep per-iteration p = {p_bms:.4f} > 1.0; reduce batch size or bands."
                )

            def make_epoch_sampler(epoch):
                return BMinSepSampler(
                    train_dataset,
                    bands=args.bands,
                    sampling_prob=p_bms,
                    n_steps=expected_steps_per_epoch,
                    key=fold_in(key(args.seed), epoch),
                )

    elif args.mechanism == "blt":
        _blt_sampler = SequentialBatchSampler(
            train_dataset,
            batch_size=args.batch_size,
        )

        def make_epoch_sampler(epoch):
            return _blt_sampler

    elif args.mechanism in ("lambda_cgd", "bisr", "bsr"):
        # BnB sampler created once — the same fixed partition is reused every
        # epoch (required by BnB privacy accounting, Lemma 3.2 of
        # Choquette-Choo et al. 2024).
        _bnb_sampler = BallsInBinsSampler(
            train_dataset,
            num_bins=expected_steps_per_epoch,
            n_steps=expected_steps_per_epoch,
            key=key(args.seed),
        )

        def make_epoch_sampler(epoch):
            return _bnb_sampler

    else:  # identity, none

        def make_epoch_sampler(epoch):
            return PoissonSampler(
                train_dataset,
                sample_rate=sample_rate,
                n_steps=expected_steps_per_epoch,
                key=fold_in(key(args.seed), epoch),
            )

    print("\nSampling:")
    print(f"  Mechanism: {args.mechanism}")
    if args.mechanism == "band_mf":
        if args.band_mf_sampling == "poisson":
            print(
                f"  Sampler: poisson (bands={args.bands}, q={sampling_prob:.6f})"
            )
        else:
            print(
                f"  Sampler: b_min_sep (bands={args.bands}, p_0={p0:.6f}, p={p_bms:.6f})"
            )
    elif args.mechanism == "blt":
        print("  Sampler: sequential (fixed order, drop_last=True)")
        print(f"  min_sep: {expected_steps_per_epoch} (= steps/epoch)")
        print(f"  max_participations: {args.num_epochs}")
    elif args.mechanism == "lambda_cgd":
        print(
            f"  Sampler: balls-in-bins (k={expected_steps_per_epoch}, fixed partition, reused across epochs)"
        )
        print(f"  DP-λCGD: λ={args.lambda_}")
    elif args.mechanism == "bisr":
        print(
            f"  Sampler: balls-in-bins (k={expected_steps_per_epoch}, fixed partition, reused across epochs)"
        )
        print(f"  BISR bandwidth: {args.bisr_bandwidth}")
    elif args.mechanism == "bsr":
        print(
            f"  Sampler: balls-in-bins (k={expected_steps_per_epoch}, fixed partition, reused across epochs)"
        )
        print(f"  BSR bandwidth: {args.bsr_bandwidth} (closed-form coefficients)")
    else:
        print(f"  Sampler: poisson (q={sample_rate:.6f})")
    print(f"  Expected batch size: {args.batch_size}")
    print(f"  Expected steps/epoch: {expected_steps_per_epoch}")

    # --- Gradient checkpointing ---
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("\nGradient checkpointing: enabled")

    offload_ctx = (
        torch.autograd.graph.save_on_cpu(pin_memory=True)
        if args.cpu_offload
        else contextlib.nullcontext()
    )

    # --- Functional conversion ---
    print("\nConverting to functional form...")
    t0 = time.time()
    fmodel, trainable_params, frozen_params = make_functional(
        model,
        disable_autograd_tracking=True,
        partition_trainable=True,
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

    if args.per_group_clipping:
        clip_norm = per_group(
            trainable_params,
            patterns=args.per_group_clipping,
            fallback=args.per_group_clipping_fallback,
        )
        print("\nPer-group clipping norms:")
        for gname, val in clip_norm.values.items():
            count = sum(1 for g in clip_norm.groups.values() if g == gname)
            print(f"  {gname}: {val:.3f} ({count} params)")
        print(f"  Effective (L2 of group bounds): {clip_norm.effective:.3f}")
    else:
        clip_norm = args.clipping_norm

    grad_fn, clip_state = clipped_grad(
        per_example_loss_fn,
        argnums=0,
        batch_argnums=(1,),
        clipping_norm=clip_norm,
        normalize_by=args.batch_size,
        microbatch_size=args.microbatch_size,
        return_aux=True,
        second_moment=args.second_moment,
    )
    zeta = (
        clip_norm.effective / args.batch_size
        if isinstance(clip_norm, PerGroup)
        else float(clip_norm) / args.batch_size
    )

    # --- Total steps & LR schedule ---
    total_steps = args.num_epochs * expected_steps_per_epoch
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    lr_schedule = make_lr_schedule(
        args.learning_rate,
        total_steps,
        warmup_frac=args.warmup_frac,
    )

    if args.warmup_frac > 0:
        print(
            f"\nLR schedule: linear warmup {args.warmup_frac:.0%} of steps → constant {args.learning_rate}"
        )
    else:
        print(f"\nLR schedule: constant {args.learning_rate} (no warmup)")
    print(f"  Peak LR: {args.learning_rate}")
    print(f"  Total steps: {total_steps}")

    # --- Privacy calibration (single-shot) ---
    if args.target_delta is None:
        args.target_delta = 1.0 / (global_train_size**1.1)

    # Build the accounting mechanism.
    #
    # Two orthogonal toggles:
    #   ``is_adam_family``: optimizer has a first-moment EMA at β₁
    #     (``adamw``, ``ademamix``, ``lion``).  Drives MF workload
    #     momentum.
    #   ``use_second_moment`` (= ``args.second_moment``): switch from
    #     single-stream MF noise to private first+second moment noise.
    #     Requires the optimizer to consume ``SecondMomentNoiseOutput``.
    is_adam_family = args.optimizer in ("adamw", "ademamix", "lion")
    use_second_moment = args.second_moment

    if use_second_moment and args.optimizer not in ("adamw", "ademamix"):
        raise ValueError(
            f"--second-moment requires an Adam-family optimizer with a second moment "
            f"to consume the paired ``SecondMomentNoiseOutput`` stream "
            f"(``adamw`` or ``ademamix``); got --optimizer {args.optimizer}.  "
            f"Run without --second-moment, or switch optimizer."
        )
    if use_second_moment and args.mechanism in ("identity", "none"):
        raise ValueError(
            "--second-moment requires a correlated MF mechanism, not identity/none."
        )

    def _workload_momentum() -> float:
        """Workload momentum for the primary (first moment) strategy."""
        return args.beta1 if is_adam_family else args.momentum

    def _make_strategy(momentum_override=None, lr_sched=None):
        """Build a strategy for the selected mechanism with given workload momentum."""
        mom = (
            momentum_override if momentum_override is not None else _workload_momentum()
        )
        if args.mechanism == "band_mf":
            return band_mf_strategy(
                n_steps=total_steps,
                bands=args.bands,
                momentum=mom,
                lr_schedule=lr_sched,
            )
        elif args.mechanism == "blt":
            return blt_strategy(
                n_steps=total_steps,
                min_sep=expected_steps_per_epoch,
                max_participations=args.num_epochs,
                max_buffers=args.max_buffers,
                momentum=mom,
                lr_schedule=lr_sched,
            )
        elif args.mechanism == "lambda_cgd":
            return lambda_cgd_strategy(
                lambda_=args.lambda_,
                n_steps=total_steps,
                min_sep=expected_steps_per_epoch,
                max_participations=args.num_epochs,
            )
        elif args.mechanism == "bisr":
            return bisr_strategy(
                bandwidth=args.bisr_bandwidth,
                n_steps=total_steps,
                min_sep=expected_steps_per_epoch,
                max_participations=args.num_epochs,
                momentum=mom,
            )
        elif args.mechanism == "bsr":
            return bsr_strategy(
                bandwidth=args.bsr_bandwidth,
                n_steps=total_steps,
                min_sep=expected_steps_per_epoch,
                max_participations=args.num_epochs,
                alpha=args.bsr_alpha,
                beta=mom,
            )
        elif args.mechanism == "identity":
            return identity_strategy()
        else:
            return None

    strategy = _make_strategy(lr_sched=lr_schedule)

    # No paired-stream wrap on the accountant: the joint Mahalanobis
    # allocation in :func:`mf_noise` makes the paired-release PLD
    # identical to the first-moment-only release at the same noise
    # multiplier (internal ``opaque._noise_allocation.paired_noise_stddevs``).

    acc.set_discretization(num_mc_samples=args.mc_samples, seed=args.seed)

    if args.mechanism == "band_mf" and strategy is not None:

        def acct_mechanism(nm):
            mechanism = ftrl_acc.band_mf(
                nm,
                sensitivity=strategy.sensitivity,
                coefficients=strategy.coefficients,
            )
            if args.band_mf_sampling == "poisson":
                return ftrl_acc.poisson(
                    mechanism, sample_rate=sampling_prob, n_steps=total_steps
                )
            return ftrl_acc.b_min_sep(
                mechanism,
                n_steps=total_steps,
                p0=p0,
            )
    elif args.mechanism == "blt" and strategy is not None:

        def acct_mechanism(nm):
            return ftrl_acc.blt(nm, sensitivity=strategy.sensitivity)
    elif args.mechanism == "lambda_cgd" and strategy is not None:

        def acct_mechanism(nm):
            mechanism = ftrl_acc.lambda_cgd(
                nm,
                sensitivity=strategy.sensitivity,
                gram_matrix=strategy.gram_matrix,
            )
            return ftrl_acc.balls_in_bins(
                mechanism,
                num_bins=expected_steps_per_epoch,
                n_steps=expected_steps_per_epoch * args.num_epochs,
            )
    elif args.mechanism == "bisr" and strategy is not None:

        def acct_mechanism(nm):
            mechanism = ftrl_acc.bisr(
                nm,
                sensitivity=strategy.sensitivity,
                gram_matrix=strategy.gram_matrix,
            )
            return ftrl_acc.balls_in_bins(
                mechanism,
                num_bins=expected_steps_per_epoch,
                n_steps=expected_steps_per_epoch * args.num_epochs,
            )
    elif args.mechanism == "bsr" and strategy is not None:

        def acct_mechanism(nm):
            mechanism = ftrl_acc.bsr(
                nm,
                sensitivity=strategy.sensitivity,
                gram_matrix=strategy.gram_matrix,
            )
            return ftrl_acc.balls_in_bins(
                mechanism,
                num_bins=expected_steps_per_epoch,
                n_steps=expected_steps_per_epoch * args.num_epochs,
            )
    elif args.mechanism == "identity":

        def acct_mechanism(nm):
            return ftrl_acc.poisson(
                ftrl_acc.mf_identity(nm),
                sample_rate=sample_rate,
                n_steps=total_steps,
            )
    elif args.mechanism == "none":

        def acct_mechanism(nm):
            return acc.nonprivate()
    else:
        raise ValueError(f"Unknown mechanism: {args.mechanism}")

    if args.mechanism == "none":
        noise_multiplier = 0.0
        print("\nNon-DP mode: noise_multiplier=0 (no privacy)")
    elif args.noise_multiplier is not None:
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
        print(
            f"  Calibrated in {time.time() - t0:.1f}s: σ={noise_multiplier:.4f} (ε≈{calibration.achieved:.3f})"
        )

    if use_wandb:
        wandb.config.update(
            {
                "noise_multiplier": noise_multiplier,
                "target_delta": args.target_delta,
                "total_steps": total_steps,
            },
            allow_val_change=True,
        )

    # --- Create MF noise function + optimizer ---
    base_noise_stddev = noise_multiplier * zeta

    if args.mechanism == "none":
        print("\nNon-DP mode: using identity noise with stddev=0...")
    elif args.mechanism == "identity":
        print("\nCreating identity noise (i.i.d. Gaussian, DP-SGD baseline)...")
    elif use_second_moment:
        print(
            f"\nCreating private second-moment noise (β₁={args.beta1}, β₂={args.beta2})..."
        )
    else:
        print(f"\nCreating MF noise (β={args.momentum})...")
    t0 = time.time()

    if use_second_moment and args.mechanism not in ("identity", "none"):
        second_strategy = _make_strategy(
            momentum_override=args.beta2, lr_sched=lr_schedule
        )
        noise_fn, noise_state = mf_noise(
            trainable_params,
            strategy,
            noise_multiplier=noise_multiplier,
            key=key(args.seed),
            second_moment_strategy=second_strategy,
        )
    elif args.mechanism in ("identity", "none"):
        noise_fn, noise_state = mf_noise(
            trainable_params,
            identity_strategy(),
            noise_multiplier=noise_multiplier,
            key=key(args.seed),
        )
    else:
        noise_fn, noise_state = mf_noise(
            trainable_params,
            strategy,
            noise_multiplier=noise_multiplier,
            key=key(args.seed),
        )
    print(f"  Noise function created in {time.time() - t0:.1f}s")

    lr_callable = lambda step: lr_schedule[min(step, len(lr_schedule) - 1)].item()  # noqa: E731

    if args.optimizer == "sgd":
        from opaque.optimizers import sgd

        optimizer = sgd(
            lr=lr_callable,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adamw":
        from opaque.optimizers import adamw

        # ``--second-moment`` drives the noise side; the same optimizer
        # consumes ``SecondMomentNoiseOutput`` when the noise output carries it.
        optimizer = adamw(
            lr=lr_callable,
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "ademamix":
        from opaque.optimizers import ademamix

        # β₃ and α default to the paper values (0.9999, 5.0).  Expose
        # CLI knobs for them once a real user case appears.
        optimizer = ademamix(
            lr=lr_callable,
            betas=(args.beta1, args.beta2, 0.9999),
            alpha=5.0,
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "lion":
        from opaque.optimizers import lion

        optimizer = lion(
            lr=lr_callable,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    opt_state = optimizer.init(trainable_params)

    # --- Diagnostic: compute what identity baseline σ would be ---
    identity_sigma = None
    if args.mechanism not in ("identity", "none") and args.noise_multiplier is None:
        try:

            def identity_acct(nm):
                return ftrl_acc.poisson(
                    ftrl_acc.mf_identity(nm),
                    sample_rate=sample_rate,
                    n_steps=total_steps,
                )

            identity_cal = cal.calibrate(
                cal.epsilon_budget(args.target_epsilon, delta=args.target_delta),
                identity_acct,
                param_min=args.calibration_min,
                param_max=args.calibration_max,
                tolerance=args.calibration_tolerance,
            )
            identity_sigma = identity_cal.param
        except Exception:
            pass  # Non-critical diagnostic

    print("\nDP-FTRL setup:")
    print(f"  Mechanism: {args.mechanism}")
    second_moment_note = " + private second moment" if use_second_moment else ""
    if is_adam_family:
        print(
            f"  Optimizer: {args.optimizer}{second_moment_note} "
            f"(β₁={args.beta1}, β₂={args.beta2}, "
            f"ε={args.adam_eps}, weight_decay={args.weight_decay})"
        )
        if use_second_moment:
            print(
                f"  Workload: EMA β₁={args.beta1} (1st moment), β₂={args.beta2} (2nd moment)"
            )
        else:
            print(f"  Workload: EMA β₁={args.beta1} (single-stream MF)")
        if args.mechanism == "bsr":
            print(
                f"  BSR workload (α={args.bsr_alpha}, β=β₁={args.beta1}): "
                "noise strategy uses paper (α,β); independent of optimizer --weight-decay; require α>β."
            )
    else:
        # sgd / lion (lion technically has β₁ but no second moment; treat like
        # the SGD-style printout since neither consumes a squared-gradient stream).
        print(
            f"  Optimizer: {args.optimizer} (β={args.momentum}, "
            f"weight_decay={args.weight_decay})"
        )
        print(
            f"  Workload: momentum-SGD (β={args.momentum})"
            f"{' [prefix-sum]' if args.momentum == 1.0 else ''}"
        )
        if args.mechanism == "bsr":
            print(
                f"  BSR workload (α={args.bsr_alpha}, β={args.momentum}): "
                "paper (α,β) for noise; optimizer --weight-decay is separate."
            )
            print(
                "  Note: BSR coefficients assume constant LR in the paper; "
                "this script still uses the LR schedule only in the optimizer."
            )
    if isinstance(clip_norm, PerGroup):
        print("  Clipping norm: per-group (fixed patterns)")
        print(
            f"  Sensitivity (effective ζ): {zeta:.6f} (= {clip_norm.effective:.3f} / batch)"
        )
    else:
        print(f"  Clipping norm: {clip_norm} (fixed)")
        print(f"  Sensitivity: {zeta:.6f} (= {clip_norm} / {args.batch_size})")
    print(f"  Noise multiplier (σ): {noise_multiplier:.4f}")
    if use_second_moment and args.mechanism not in ("identity", "none"):
        # Per-stream σ is allocated by the dispatcher
        # (sensitivity-proportional Mahalanobis on the encoded streams) and
        # surfaces at runtime via ``noise_output.{noisy_grads,noisy_squared_grads}.noise_stddev``.
        print("  Paired-stream allocation: sensitivity-proportional Mahalanobis")
    else:
        print(
            f"  Base noise stddev: {base_noise_stddev:.6f} (= {noise_multiplier:.4f} × {zeta:.6f})"
        )
    if identity_sigma is not None:
        ratio = noise_multiplier / identity_sigma
        print(f"  Identity baseline σ: {identity_sigma:.4f} (ratio: {ratio:.2f}×)")
        print(
            f"  → MF needs {ratio:.2f}× more noise to hit ε={args.target_epsilon}; "
            f"correlated structure must compensate"
        )
    print(f"  Microbatch size: {args.microbatch_size}")
    if args.mechanism == "band_mf":
        print(f"  Bands: {args.bands}")
    elif args.mechanism == "blt":
        print(f"  Max buffers: {args.max_buffers}")
        print(f"  Min separation: {expected_steps_per_epoch}")
        print(f"  Max participations: {args.num_epochs}")
    elif args.mechanism == "lambda_cgd":
        print(f"  λ (lambda): {args.lambda_}")
        print("  Bandwidth: 2 (bidiagonal inverse)")
        print("  Column normalization: enabled (Appendix A, exact BnB)")
    elif args.mechanism == "bisr":
        print(f"  BISR bandwidth: {args.bisr_bandwidth}")
        print("  Column normalization: enabled (Appendix A, exact BnB)")
    elif args.mechanism == "bsr":
        bsr_beta = args.beta1 if is_adam_family else args.momentum
        print(
            f"  BSR: bandwidth={args.bsr_bandwidth}, workload (α={args.bsr_alpha}, β={bsr_beta})"
        )

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
        wandb.log(
            {"eval/loss": initial_eval_loss, "train/lr": lr_schedule[0].item()}, step=0
        )

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 80)

        epoch_loader = DataLoader(
            train_dataset,
            batch_sampler=make_epoch_sampler(epoch),
            collate_fn=collate,
        )

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
                        trainable_params,
                        input_ids,
                        state=clip_state,
                    )

                # ``grads`` is a ``SecondMomentClippingOutput`` when
                # ``--second-moment`` is on (clipped_grad produced both
                # streams per-example), or a single ``ClippedPytree``
                # otherwise — the noise function dispatches polymorphically.
                noisy_grads, noise_state = noise_fn(grads, noise_state)
                if isinstance(noisy_grads, SecondMomentNoiseOutput):
                    step_noise_stddev = noisy_grads.noisy_grads.noise_stddev
                else:
                    step_noise_stddev = noisy_grads.noise_stddev
                updates, opt_state = optimizer.update(
                    noisy_grads,
                    opt_state,
                    params=trainable_params,
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
                    wandb.log(
                        {
                            "train/loss": avg_loss,
                            "train/batch_size": batch_size,
                            "train/clipping_norm": (
                                clip_norm.effective
                                if isinstance(clip_norm, PerGroup)
                                else clip_norm
                            ),
                            "train/clip_rate": clip_rate,
                            "train/grad_norm_mean": mean_grad_norm,
                            "train/noise_std": step_noise_stddev,
                            "train/lr": lr_t,
                            "train/momentum": args.momentum,
                            "perf/step_time_sec": perf["step_time_sec"],
                            "perf/throughput_samples_per_sec": perf[
                                "throughput_samples_sec"
                            ],
                            "perf/peak_gb": perf["memory_peak_gb"],
                        },
                        step=global_step,
                    )

                print(
                    f"Step {global_step:4d} [E{epoch + 1} S{step_idx + 1:3d}/{expected_steps_per_epoch:3d}] | "
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
    second_moment_suffix = " + private second moment" if use_second_moment else ""
    if is_adam_family:
        print(
            f"Optimizer: {args.optimizer}{second_moment_suffix} "
            f"(β₁={args.beta1}, β₂={args.beta2})"
        )
    else:
        print(f"Optimizer: {args.optimizer} (β={args.momentum})")
    print("\nTraining results:")
    print(f"  Total steps: {global_step}")
    if losses:
        print(f"  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
    if clip_rates:
        print("\nClipping:")
        if isinstance(clip_norm, PerGroup):
            print(f"  Per-group clip (effective): {clip_norm.effective:.3f}")
        else:
            print(f"  Fixed norm: {clip_norm}")
        print(f"  Average clip rate: {sum(clip_rates) / len(clip_rates):.2%}")

    # Single-shot accounting
    if args.mechanism == "none":
        print("\nPrivacy: Non-DP baseline (no privacy guarantee)")
    else:
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
