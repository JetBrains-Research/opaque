"""End-to-end DP-SGD LoRA training example for causal language models.

This example is designed as a production-style script (not a tutorial):
- clipping + noise + accounting always enabled
- auto clipping by default (--clipping-mode fixed|adaptive|auto)
- noise multiplier calibrated from target privacy budget
- privacy and grad-norm telemetry reported every eval_steps
- optional empirical privacy auditing with W&B integration

USAGE:

  # Quick smoke test (~5 minutes, GPT-2 on ag_news)
  python examples/train_causal_lm.py --preset smoke

  # Or use default settings (same as smoke)
  python examples/train_causal_lm.py

  # Full production training on Mellum-4b + KStack (~3-5 hours)
  python examples/train_causal_lm.py --preset mellum-kstack

  # 4-GPU distributed run with torchrun
  torchrun --nproc_per_node=4 examples/train_causal_lm.py --preset mellum-kstack

  # Or customize individual parameters:
  python examples/train_causal_lm.py \\
    --model-name "JetBrains/Mellum-4b-base" \\
    --dataset "JetBrains/KStack" \\
    --dataset-text-field "content" \\
    --num-train-samples 50000 \\
    --num-eval-samples 1000 \\
    --num-epochs 3 \\
    --batch-size 32 \\
    --eval-steps 50 \\
    --target-epsilon 10.0 \\
    --learning-rate 5e-5 \\
    --lora-r 16 --lora-alpha 32 \\
    --max-seq-len 1024 \\
    --lora-modules q_proj k_proj v_proj o_proj \\
    --audit --audit-canaries 1000 \\
    --no-wandb
"""

import argparse
import contextlib
import importlib.util
import itertools
import math
import os
import sys
import time

import torch
import torch.distributed as dist
import torchopt
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from lora_privacy.peft_lora_xs import LoraXSConfig

from opaque.patches import apply_model_patches, apply_runtime_patches

apply_runtime_patches()
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

# examples/ is sys.path[0] when this file is run as a script, so a bare import
# resolves. Kept as a sibling module rather than inlined because the trainer
# cannot run on macOS and glue_data is the only part of the classification path
# that can be tested locally -- see examples/test_glue_data.py.
import glue_data

import opaque.accounting as acc
import opaque.auditing as auditing
import opaque.dpsgd.accounting as dpsgd_acc
from opaque.accounting import calibration as cal, Accountant
from opaque.clipping import clipped_grad
from opaque.dpsgd.clipping import adaptive_clipped_grad, auto_clipped_grad
from opaque.distributed import sync
from opaque.distributed.gradients import sum_gradients_
from opaque.dpsgd.noise import gaussian_noise
from opaque.dpsgd.noise import per_group_noise_stddev
from opaque.dpsgd.noise import truncated_gaussian_noise
from opaque.profiling import (
    StepTimer,
    TrainingProfiler,
    print_memory,
    reset_peak_memory,
)
from opaque.random import key, fold_in
from opaque.dpsgd.sampling import PoissonSampler
from opaque.dpsgd.sampling import TruncatedPoissonSampler
from opaque.distributed import local_shard
from opaque.functional import make_functional
from opaque.scheduling import (
    cosine_schedule,
    inverse_sqrt_schedule,
    linear_schedule,
    with_warmup,
)
from opaque.scheduling.types import Schedule
from opaque.types import PerGroup, SecondMomentClippingOutput, SecondMomentNoiseOutput
from opaque.clipping import per_group
import wandb


def _effective(value):
    """Extract scalar from float or PerGroup for logging/printing."""
    return value.effective if isinstance(value, PerGroup) else value


def _noise_stddev(max_norm, noise_multiplier, *, per_group=True):
    """Noise stddev: MSE-optimal per-group when available, isotropic otherwise."""
    if per_group and isinstance(max_norm, PerGroup):
        return per_group_noise_stddev(max_norm, noise_multiplier)
    return noise_multiplier * max_norm


def _step_clip_norm(grads_tuple):
    """``.max_norm`` from a clipped pytree, unwrapping the paired SM output."""
    if isinstance(grads_tuple, SecondMomentClippingOutput):
        return grads_tuple.grads.max_norm
    return grads_tuple.max_norm


def _step_noise_stddev(noisy_grads):
    """``.noise_stddev`` from a noised pytree, unwrapping the paired SM output."""
    if isinstance(noisy_grads, SecondMomentNoiseOutput):
        return noisy_grads.noisy_grads.noise_stddev
    return noisy_grads.noise_stddev


def _select_device(local_rank: int | None = None) -> tuple[torch.device, str]:
    """Select best available device with user-facing label."""
    if torch.cuda.is_available():
        if local_rank is not None:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            return device, torch.cuda.get_device_name(local_rank)
        device = torch.device("cuda")
        return device, torch.cuda.get_device_name(0)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        return device, "Apple Silicon"

    return torch.device("cpu"), "CPU"


def _init_distributed() -> tuple[bool, int, int, int]:
    """Initialize torch.distributed when launched via torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 1:
        return False, 0, 1, 0

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    return True, rank, world_size, local_rank


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _is_dtype_supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Check whether a dtype can be allocated on a specific device."""
    try:
        torch.empty((1,), device=device, dtype=dtype)
        return True
    except (RuntimeError, TypeError):
        return False


def _resolve_model_dtype(
    requested_name: str,
    device: torch.device,
) -> tuple[str, torch.dtype, str | None]:
    """Resolve requested dtype with safe fallback for unsupported device paths."""
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
            reason = (
                f"Requested dtype '{requested_name}' is not supported on {device.type}; "
                f"using '{fallback_name}' instead."
            )
            return fallback_name, fallback_dtype, reason

    raise RuntimeError(
        f"No supported dtype found for device={device.type}. "
        f"Requested dtype was '{requested_name}'."
    )


def _kernel_mode_summary(device: torch.device, dtype_name: str) -> tuple[str, str]:
    """Return concise status of kernel optimization mode for this run."""
    if os.environ.get("OPAQUE_NO_PATCH", "0") == "1":
        return "disabled", "OPAQUE_NO_PATCH=1"

    if os.environ.get("OPAQUE_NO_KERNEL_PATCH", "0") == "1":
        return "disabled", "OPAQUE_NO_KERNEL_PATCH=1"

    if device.type != "cuda":
        return "disabled", f"device={device.type} (Triton kernels are CUDA-only)"

    if importlib.util.find_spec("triton") is None:
        return "disabled", "triton package not installed"

    if dtype_name not in {"float16", "bfloat16"}:
        return "partial", f"dtype={dtype_name} (fused CE requires fp16/bf16)"

    return "enabled", "explicit kernel patches applied"


def _print_runtime_mode_report(
    device: torch.device,
    device_label: str,
    dtype_name: str,
    dtype: torch.dtype,
    dtype_warning: str | None,
) -> None:
    """Print active runtime mode so fallback behavior is explicit."""
    kernel_mode, kernel_reason = _kernel_mode_summary(device, dtype_name)

    print("\nRuntime mode:")
    print(f"  Device: {device} ({device_label})")
    print(f"  Dtype: {dtype_name} ({dtype})")
    if dtype_warning:
        print(f"  Dtype fallback: {dtype_warning}")
    print(f"  Kernel optimizations: {kernel_mode} ({kernel_reason})")

    if device.type == "cpu":
        print("  Note: CPU path prioritizes correctness over throughput.")
    elif device.type == "mps":
        print(
            "  Note: MPS uses compatibility fallbacks when CUDA-only kernels are unavailable."
        )


def _load_streaming_subset(
    dataset_name: str,
    dataset_subset: str | None,
    dataset_split: str,
    dataset_text_field: str,
    total_needed: int,
) -> Dataset:
    """Stream only required rows, then materialize to in-memory static Dataset."""
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
            f"Text field '{dataset_text_field}' not found in streamed sample. "
            f"Available fields: {list(rows[0].keys())}"
        )

    if len(rows) < total_needed:
        raise ValueError(
            f"Stream ended after {len(rows)} examples, but {total_needed} are required "
            f"(train + eval)."
        )

    return Dataset.from_list(rows)


def parse_args():
    """Parse command-line arguments with logical groups."""
    parser = argparse.ArgumentParser(
        description="End-to-end DP-SGD LoRA training for Causal Language Models"
    )

    # Preset configurations
    parser.add_argument(
        "--preset",
        type=str,
        choices=[
            "custom",
            "smoke",
            "mellum-kstack",
            "qwen-7b-kstack",
            "qwen-coder-kstack-lora",
            "roberta-large-glue",
        ],
        default="smoke",
        help=(
            "Apply preset configuration (custom=keep explicit args, "
            "smoke=quick test ~2min, mellum-kstack=Mellum-4b + KStack at ε=10 "
            "with adafactor @ 5e-5, qwen-7b-kstack=Qwen2.5-Coder-7B + KStack at "
            "ε=3 with adafactor @ 5e-4, qwen-coder-kstack-lora=tuned vanilla LoRA "
            "baseline for the same model/dataset with SGD, roberta-large-glue="
            "the LoRA-XS paper's GLUE setup so our numbers sit in their Table 1)."
        ),
    )

    model_group = parser.add_argument_group("model", "Model and tokenizer settings")
    model_group.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
        help="HuggingFace model name or local path",
    )
    model_group.add_argument(
        "--classifier-lr",
        type=float,
        default=None,
        help=(
            "Separate learning rate for the randomly-initialised classification "
            "head, for --task-type sequence-classification. Defaults to "
            "--learning-rate. The LoRA-XS paper tunes this independently per task "
            "and rank (Table 7): at r=16 it is 1e-2 for CoLA against an adapter "
            "lr of 1e-3, i.e. 10x. Without it the head trains at the adapter's "
            "rate, and the head's gradients dominate -- measured at 95.6 mean "
            "norm on GLUE against 0.30 on KStack, where every trainable is a "
            "LoRA-XS core started at sigma=1e-5 on a converged model."
        ),
    )
    model_group.add_argument(
        "--task-type",
        type=str,
        choices=["causal-lm", "sequence-classification"],
        default="causal-lm",
        help=(
            "What to train. 'causal-lm' is the default next-token objective. "
            "'sequence-classification' loads AutoModelForSequenceClassification "
            "and trains on a GLUE task selected with --glue-task, which is how "
            "our numbers become comparable to the LoRA-XS paper's Table 1."
        ),
    )
    model_group.add_argument(
        "--attention",
        type=str,
        choices=["eager", "sdpa"],
        default="sdpa",
        help="Attention implementation (default: sdpa, which is faster and uses less memory)",
    )
    model_group.add_argument(
        "--sdpa-backend",
        type=str,
        choices=["flash", "efficient", "cudnn", "math"],
        default=None,
        help="Force a specific SDPA backend (default: None = PyTorch auto-selects)",
    )

    data_group = parser.add_argument_group("data", "Dataset and tokenization settings")
    data_group.add_argument(
        "--lr-warmup-ratio",
        type=float,
        default=None,
        help=(
            "Warmup as a fraction of total steps, resolved once the dataset size "
            "is known; overrides --lr-warmup-steps when set. The LoRA-XS paper "
            "uses 0.06 on GLUE. Worth having as a ratio and not a step count "
            "because GLUE task sizes differ by 3x, and zero warmup under AdamW is "
            "exactly what blew up the full-LoRA baseline (ref-lora-r16-adamw-lr1e3)."
        ),
    )
    data_group.add_argument(
        "--glue-task",
        type=str,
        default=None,
        help=(
            "GLUE task for --task-type sequence-classification: cola, sst2, mrpc, "
            "stsb, qnli, rte, mnli, qqp. Required for that task type and ignored "
            "otherwise. --dataset/--dataset-split/--dataset-text-field are all "
            "unused on this path: the split is GLUE's own `validation`, because "
            "that is what every published number is computed on."
        ),
    )
    data_group.add_argument(
        "--dataset", type=str, default="ag_news", help="HuggingFace dataset name"
    )
    data_group.add_argument(
        "--dataset-subset",
        "--dataset-name",
        dest="dataset_subset",
        type=str,
        default=None,
        help="Optional dataset subset (HF load_dataset 'name' argument), e.g. 'stage1-auto-format'.",
    )
    data_group.add_argument(
        "--dataset-split", type=str, default="train", help="Dataset split for training"
    )
    data_group.add_argument(
        "--dataset-text-field",
        type=str,
        default="text",
        help="Field containing text",
    )
    data_group.add_argument(
        "--num-train-samples",
        type=int,
        default=5000,
        help="Number of training examples (default: 5000 for smoke test)",
    )
    data_group.add_argument(
        "--num-eval-samples",
        "--num-eval-samples-alt",
        dest="num_eval_samples",
        type=int,
        default=100,
        help="Number of samples for periodic eval-loss reporting (batched)",
    )
    data_group.add_argument(
        "--max-seq-len", type=int, default=512, help="Maximum sequence length"
    )

    train_group = parser.add_argument_group("training", "Training loop settings")
    train_group.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Expected batch size for Poisson sampling (determines sample_rate)",
    )
    train_group.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Batch size for evaluation (default: same as batch_size, can be larger since no privacy needed)",
    )
    train_group.add_argument(
        "--num-epochs", type=int, default=3, help="Number of epochs"
    )
    train_group.add_argument(
        "--learning-rate", type=float, default=1.0e-5, help="Learning rate"
    )
    train_group.add_argument(
        "--lr-schedule",
        type=str,
        default="none",
        choices=["none", "cosine", "linear", "sqrt"],
        help=(
            "LR schedule applied to ``--learning-rate``.  ``cosine`` decays "
            "from peak to ``lr * lr_min_ratio`` over the full training run; "
            "``linear`` decays linearly to the same floor; ``sqrt`` decays "
            "as 1/sqrt(1 + step/warmup) (Adagrad-mimic).  Default ``none`` "
            "keeps the constant LR."
        ),
    )
    train_group.add_argument(
        "--lr-min-ratio",
        type=float,
        default=0.0,
        help="Floor LR as a fraction of peak (cosine/linear).  Default 0 (decay to zero).",
    )
    train_group.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=0,
        help="Linear warmup from 0 to peak LR over this many steps (any schedule).",
    )
    train_group.add_argument(
        "--optimizer",
        type=str,
        default="adafactor",
        choices=[
            "sgd",
            "adam",
            "adamw",
            "ademamix",
            "lion",
            "adafactor",
            "rmsprop",
            "adagrad",
        ],
        help=(
            "Optimizer.  ``sgd`` and ``adam`` are torchopt's vanilla "
            "primitives (no DP-aware paths); the others are Opaque-built "
            "(see opaque.optimizers).  Pair with "
            "``--noise-bias-correction`` to enable DP-aware bias "
            "correction where applicable."
        ),
    )
    train_group.add_argument(
        "--noise-bias-correction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable DP noise-variance bias correction on optimizers that "
            "support it (adam/adamw/ademamix/rmsprop/adagrad/adafactor).  "
            "Silently ignored on sgd/lion.  Off by default; see "
            "docs/user-guide/optimizers.md for when to flip it on."
        ),
    )
    train_group.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizers that support it (default: 0.01)",
    )
    train_group.add_argument(
        "--log-steps",
        type=int,
        default=1,
        help="Log training metrics every N steps",
    )
    train_group.add_argument(
        "--eval-steps",
        type=int,
        default=10,
        help="Log eval loss and privacy every N steps",
    )
    train_group.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps (overrides num_epochs if set)",
    )
    train_group.add_argument("--seed", type=int, default=42, help="Random seed")
    train_group.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable gradient checkpointing for memory savings (trades compute for memory)",
    )
    train_group.add_argument(
        "--cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Offload saved tensors to CPU via save_on_cpu (works with or without checkpointing)",
    )
    train_group.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Linear LR warmup steps (0 = no warmup)",
    )
    train_group.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Save model adapter to this directory after training (enables downstream eval)",
    )
    train_group.add_argument(
        "--eval-humaneval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run HumanEval evaluation after training (requires --output-dir)",
    )
    train_group.add_argument(
        "--eval-humaneval-n-samples",
        type=int,
        default=164,
        help="Number of HumanEval problems to evaluate (default: 164 = all)",
    )
    train_group.add_argument(
        "--eval-mbpp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run MBPP+ evaluation after training (requires --output-dir)",
    )
    train_group.add_argument(
        "--restore-best-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Before saving / downstream eval, restore the trainable params from "
            "the step with the lowest eval/loss instead of using the final step. "
            "Removes the single-checkpoint lottery from downstream pass@1. "
            "Tracked params are snapshotted to CPU at each new-best eval."
        ),
    )
    train_group.add_argument(
        "--eval-ema-beta",
        type=float,
        default=0.7,
        help=(
            "EMA decay for the smoothed eval/loss metric (eval/loss_ema). "
            "0 disables EMA logging. The smoothed metric makes run-to-run "
            "comparison robust to per-checkpoint eval noise."
        ),
    )

    lora_group = parser.add_argument_group("lora", "LoRA adapter settings")
    lora_group.add_argument(
        "--lora-method",
        type=str,
        choices=["lora", "lora-xs"],
        default="lora",
        help="LoRA variant: lora (standard) or lora-xs (SVD factors + trainable r×r R matrix)",
    )
    lora_group.add_argument("--lora-r", type=int, default=4, help="LoRA rank")
    lora_group.add_argument("--lora-alpha", type=int, default=8, help="LoRA alpha")
    lora_group.add_argument(
        "--lora-modules",
        type=str,
        nargs="+",
        default=["c_attn", "c_proj"],
        help="Target module names for LoRA",
    )
    lora_group.add_argument(
        "--lora-dora",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use DoRA (weight-decomposed LoRA). Only applies to --lora-method lora.",
    )
    lora_group.add_argument(
        "--lora-rslora",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use rank-stabilized scaling (alpha/sqrt(r) instead of alpha/r). Only applies to --lora-method lora.",
    )
    lora_group.add_argument(
        "--lora-init",
        type=str,
        default="default",
        choices=["default", "gaussian", "pissa", "pissa_niter_4", "olora", "loftq"],
        help="LoRA weight initialization strategy (default: Kaiming for A, zero for B). "
        "Only applies to --lora-method lora.",
    )
    lora_group.add_argument(
        "--lora-xs-sigma",
        type=float,
        default=1e-5,
        help="LoRA-XS: R matrix init std N(0, sigma^2) (default: 1e-5)",
    )
    lora_group.add_argument(
        "--lora-xs-rank-pattern-json",
        type=str,
        default=None,
        help=(
            "LoRA-XS: path to a JSON dict {module_name_or_pattern: r_l} of "
            "per-layer rank overrides (variable-rank / Rényi allocation). "
            "Modules not listed use --lora-r. Generate it with "
            "examples/compute_rank_allocation.py."
        ),
    )
    lora_group.add_argument(
        "--lora-xs-rank-pattern-b64",
        type=str,
        default=None,
        help=(
            "LoRA-XS: base64-encoded JSON {module: r_l} rank pattern, passed "
            "inline (avoids needing a file in the image). Used for probe-based "
            "allocation: probe dumps spectra -> compute allocation -> pass here."
        ),
    )
    lora_group.add_argument(
        "--lora-xs-rank-alloc",
        choices=["none", "w0"],
        default="none",
        help=(
            "LoRA-XS: in-process per-layer rank allocation. 'w0' scores each "
            "target W0 by its Rényi effective rank (--lora-xs-alloc-alpha) and "
            "allocates the fixed budget n_layers*r^2 proportionally (data-free, "
            "epsilon-free). 'none' = uniform --lora-r."
        ),
    )
    lora_group.add_argument(
        "--lora-xs-alloc-alpha",
        type=str,
        default="inf",
        help="Rényi order for --lora-xs-rank-alloc scoring ('inf'=stable rank; 1=Shannon).",
    )
    lora_group.add_argument(
        "--lora-xs-alloc-probe-r",
        type=int,
        default=32,
        help="Top singular values per layer used to score W0 for allocation.",
    )
    train_group.add_argument(
        "--eval-bpb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compute per-example bits-per-byte (NLL / UTF-8 bytes) on the eval set "
            "at the end of training. Far higher SNR than aggregate eval loss "
            "(Signal-and-Noise, arXiv 2508.13144: ~20x on code) and yields "
            "per-example values for paired bootstrap tests."
        ),
    )
    train_group.add_argument(
        "--eval-bpb-samples", type=int, default=512,
        help="Number of eval examples to score for BPB (default 512).",
    )
    train_group.add_argument(
        "--eval-bpb-microbatch", type=int, default=2,
        help="Micro-batch for BPB scoring (logits are large; keep small).",
    )
    lora_group.add_argument(
        "--dump-core-spectra",
        type=str,
        default=None,
        help=(
            "LoRA-XS: after training, dump each layer's core-R singular values "
            "to this JSON path ({module: [sigmas]}). Run a short probe with "
            "--max-steps N and this flag, then feed to compute_rank_allocation.py."
        ),
    )
    lora_group.add_argument(
        "--lora-xs-orthonormal-a",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="LoRA-XS: use A=U^T without singular values (eliminates gradient amplification under DP-SGD)",
    )
    train_group.add_argument(
        "--adam-beta2",
        type=float,
        default=0.99,
        help=(
            "Adam/AdamW second-moment decay. Applies to BOTH the plain adamw path "
            "and xse_adamw, which previously disagreed: opaque.optimizers.adamw "
            "defaulted to 0.999 while xse_adamw hard-coded 0.99, so frozen and "
            "rotating AdamW arms were not comparable. Default 0.99. "
            "The quantity that matters when rotating is tau*(1-beta2), the rotation "
            "interval divided by Adam's second-moment timescale 1/(1-beta2): below 1 "
            "the basis is rewritten before Adam can estimate a direction's typical "
            "gradient size. At tau=1 and beta2=0.99 it is 0.01."
        ),
    )
    lora_group.add_argument(
        "--lora-xs-init",
        type=str,
        choices=["weight", "grad", "grad-sb"],
        default="weight",
        help=(
            "Which basis the frozen LoRA-XS factors are built from. "
            "weight = SVD of W0 (the LoRA-XS default). "
            "grad = SVD of the first full-weight gradient, i.e. LoRA-SB's basis "
            "(arXiv:2411.19557), with R left at its usual small-Gaussian init so "
            "the only change vs weight is the subspace. "
            "grad-sb = full LoRA-SB: also seeds R = diag(S)*lr/scaling, baking "
            "the first step into the initialization."
        ),
    )
    lora_group.add_argument(
        "--lora-xs-init-batches",
        type=int,
        default=1,
        help=(
            "Batches to accumulate the gradient over for --lora-xs-init grad*. "
            "LoRA-SB uses ~1/1000 of the dataset; more batches means a less noisy "
            "basis at ~1%% of an epoch per batch."
        ),
    )
    lora_group.add_argument(
        "--lora-xs-manifold-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "LoRA-XS: parametrize R as Cayley(S)·diag(sigma) instead of dense r×r. "
            "Reduces trainable params per layer from r^2 to r(r+1)/2 (~half), giving "
            "sqrt(2)x DP-SNR boost. Combine with --per-group-clipping to apply "
            "different clipping norms to S (direction) vs sigma (magnitude)."
        ),
    )
    lora_group.add_argument(
        "--lora-xs-oft-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "LoRA-XS: parametrize R as Cayley(X)·diag(s) − I (Spectral-OFT-XS). "
            "Same param count as --lora-xs-manifold-mode (r(r+1)/2) but with "
            "s initialised near 1 instead of 0, which fixes the cold-start "
            "vanishing-gradient problem that hurt vanilla manifold mode. "
            "Mutually exclusive with --lora-xs-manifold-mode. See "
            "vendor/lora-privacy/docs/beyond-lora-research-program.md."
        ),
    )
    lora_group.add_argument(
        "--lora-xs-manifold-init-sigma-w",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Manifold-mode only: initialise sigma at W₀'s top-r singular "
            "values instead of small Gaussian. Fixes vanishing-gradient-on-"
            "skew at init by giving σ a unit-scale magnitude, so ∂R/∂S = "
            "2X·diag(σ)|_{X=0} = O(σ_W) is well-scaled. See "
            "vendor/lora-privacy/docs/oft-postmortem.md §N3."
        ),
    )
    # LoRA-XSe: exploration via momentum SVD rotation
    lora_group.add_argument(
        "--lora-xse-rotation-warmup-steps",
        type=int,
        default=0,
        help=(
            "Train this many steps before the FIRST rotation. Rotation keeps the "
            "top directions of R's momentum, so it needs a roughly stationary "
            "objective; a randomly-initialised classification head co-training at "
            "10x the adapter lr breaks that, and all four rotating CoLA arms lost "
            "to frozen. Delaying rotation lets the head settle and momentum "
            "concentrate first. 0 = rotate from the start (every run before "
            "2026-08-21). Only used when --lora-xse-p-e > 0."
        ),
    )
    lora_group.add_argument(
        "--lora-xse-p-e",
        type=float,
        default=0.0,
        help=(
            "LoRA-XSe exploration fraction (0 = plain LoRA-XS, 1/3 = default XSe). "
            "Controls what fraction of the rank is re-randomized at each rotation. "
            "Requires --optimizer sgd with --sgd-momentum > 0."
        ),
    )
    lora_group.add_argument(
        "--lora-xse-rotation-step-interval",
        type=int,
        default=None,
        help=(
            "Steps between rotations (default: auto from momentum = max(1, round(0.5/(1-β)))). "
            "Only used when --lora-xse-p-e > 0."
        ),
    )
    lora_group.add_argument(
        "--sgd-momentum",
        type=float,
        default=0.0,
        help="Momentum for --optimizer sgd (also used by xse_sgd). Default 0.0.",
    )

    dp_group = parser.add_argument_group("dp", "DP-SGD clipping and noise")
    dp_group.add_argument(
        "--shard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shard dataset across DDP ranks (default). "
        "Use --no-shard to replicate full dataset on each rank "
        "(uses parallel_poisson accounting).",
    )
    dp_group.add_argument(
        "--clipping-mode",
        type=str,
        choices=["fixed", "adaptive", "auto"],
        default="auto",
        help="Clipping strategy: fixed (constant threshold), adaptive (Andrew "
        "et al. quantile tracking), or auto (AUTO-S automatic scaling, Bu et "
        "al. NeurIPS 2023).",
    )
    dp_group.add_argument(
        "--clipping-norm",
        type=float,
        default=1.0,
        help="Clipping norm: fixed threshold C (fixed mode), starting threshold "
        "(adaptive mode), or sensitivity max_norm R (auto mode).",
    )
    dp_group.add_argument(
        "--target-clipping-rate",
        type=float,
        default=0.5,
        help="Target clipping rate for --clipping-mode adaptive.",
    )
    dp_group.add_argument(
        "--clipping-norm-max",
        type=float,
        default=10.0,
        help="Maximum clipping norm for --clipping-mode adaptive.",
    )
    dp_group.add_argument(
        "--auto-clipping-gamma",
        type=float,
        default=0.01,
        help="Denominator stabilizer γ for --clipping-mode auto "
        "(AUTO-S: g̃ = R·g / (‖g‖ + γ); default 0.01).",
    )
    dp_group.add_argument(
        "--microbatch-size",
        type=int,
        default=None,
        help="Microbatch size passed to clipped_grad/adaptive_clipped_grad (None=process full batch with vmap, faster but more memory; use 0 on CLI to mean None)",
    )
    dp_group.add_argument(
        "--sampler",
        type=str,
        choices=["poisson", "truncated_poisson"],
        default="poisson",
        help="Sampling strategy: poisson (standard, variable batch size) "
        "or truncated_poisson (batch capped at --max-batch-size for clipped memory)",
    )
    dp_group.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help="Max batch size for truncated_poisson sampler (default: same as --batch-size). "
        "Ignored for standard poisson.",
    )
    dp_group.add_argument(
        "--noise-mechanism",
        type=str,
        choices=["gaussian", "truncated_gaussian"],
        default="gaussian",
        help="Noise mechanism: gaussian (standard, unclipped) "
        "or truncated_gaussian (renormalized, clipped support)",
    )
    dp_group.add_argument(
        "--noise-radius",
        type=float,
        default=3.0,
        help="Support half-width in sigma units for rectified/truncated Gaussian (ignored for standard gaussian)",
    )
    dp_group.add_argument(
        "--second-moment",
        type=str,
        default="none",
        help="[experimental] Release a private squared-gradient stream alongside "
        "gradients (Kalinin et al., arXiv:2502.06597).  'none' (default; "
        "standard single-stream release), 'auto' (enable for optimizers with "
        "a noisy_squared_grads branch — adam/adamw/ademamix/rmsprop/radam/"
        "adadelta), or an explicit float >1.0 for the first-moment overhead "
        "(default sqrt(3/2) ≈ 1.225 when enabled).  Not yet supported with "
        "--per-group-clipping (joint first+second-moment privacy allocation "
        "has not been validated for per-group sensitivities).",
    )
    dp_group.add_argument(
        "--per-group-clipping",
        type=str,
        nargs="+",
        default=None,
        metavar="PATTERN=NORM",
        help="Per-group clipping norms as PATTERN=NORM pairs (e.g., self_attn=1.0 mlp=2.0). "
        "Each trainable param must match exactly one pattern substring. "
        "Use 'fallback=NORM' as catch-all for unmatched params. "
        "Compatible with all --clipping-mode values (adaptive adapts each group "
        "independently; auto uses per-group R_k).",
    )

    # Model precision
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model precision (default: bfloat16 for best performance/memory tradeoff)",
    )

    privacy_group = parser.add_argument_group(
        "privacy", "Privacy accounting and noise calibration"
    )
    privacy_group.add_argument(
        "--target-epsilon",
        type=float,
        default=3.0,
        help="Target epsilon used to calibrate noise_multiplier",
    )
    privacy_group.add_argument(
        "--target-delta",
        type=float,
        default=None,
        help="Target delta for DP accounting. Default: 1/n² where n = training set size.",
    )
    privacy_group.add_argument(
        "--noise-multiplier",
        type=float,
        default=None,
        help="Use a fixed noise multiplier instead of calibrating from target_epsilon",
    )
    privacy_group.add_argument(
        "--calibration-min",
        type=float,
        default=0.11,
        help="Lower bound for noise calibration search",
    )
    privacy_group.add_argument(
        "--calibration-max",
        type=float,
        default=3.5,
        help="Upper bound for noise calibration search",
    )
    privacy_group.add_argument(
        "--calibration-tolerance",
        type=float,
        default=1e-3,
        help="Tolerance for noise calibration",
    )

    audit_group = parser.add_argument_group("audit", "Empirical privacy auditing")
    audit_group.add_argument(
        "--audit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable empirical auditing (disabled by default)",
    )
    audit_group.add_argument(
        "--audit-canaries",
        type=int,
        default=1000,
        help="Number of canaries for one-run auditing",
    )
    audit_group.add_argument(
        "--audit-batch-size",
        type=int,
        default=None,
        help="Batch size for auditing scoring (default: same as microbatch_size; forward-only so less memory than training)",
    )

    tracking_group = parser.add_argument_group("tracking", "Experiment tracking (W&B)")
    tracking_group.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable experiment tracking (wandb is enabled by default, offline if no credentials)",
    )
    tracking_group.add_argument(
        "--wandb-project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "opaque"),
        help="W&B project name (default: WANDB_PROJECT env var or 'opaque')",
    )
    tracking_group.add_argument(
        "--wandb-run-name",
        type=str,
        default=os.environ.get("WANDB_NAME") or os.environ.get("RUN_NAME"),
        help="Run name (default: WANDB_NAME, then RUN_NAME env var, or auto-generated from model and hyperparameters)",
    )
    tracking_group.add_argument(
        "--wandb-entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY"),
        help="W&B entity/team (default: WANDB_ENTITY env var)",
    )

    args = parser.parse_args()

    # Track which options were explicitly provided on CLI.
    # This avoids ambiguity when an explicit value equals the parser default.
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

    # Apply preset configurations (CLI args take precedence)
    if args.preset == "smoke":
        # Quick smoke test with GPT-2 (~100 steps, ~2-3 minutes)
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
        _set("learning_rate", 1e-5)
        _set("lora_r", 4)
        _set("lora_alpha", 8)
        _set("max_seq_len", 512)
        _set("lora_modules", ["c_attn", "c_proj"])
        _set("dtype", "bfloat16")
        _set("audit", False)
    elif args.preset == "mellum-kstack":
        # Golden configuration for Mellum-4b + KStack training on H200
        # Memory: Model=7.5 GiB. Throughput saturates at mb=16 (~20 samples/s, 58 GB peak).
        # mb=32 gives same speed but 108 GB. With --gradient-checkpointing, mb=32+ fits easily.
        _set("model_name", "JetBrains/Mellum-4b-base")
        _set("dataset", "JetBrains/KStack")
        _set("dataset_text_field", "content")
        _set("num_train_samples", 50000)
        _set("num_eval_samples", 1000)
        _set("num_epochs", 3)
        _set("batch_size", 128)
        _set("log_steps", 2)
        _set("eval_steps", 10)
        _set("target_epsilon", 10.0)
        _set("learning_rate", 5e-5)
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
    elif args.preset == "qwen-7b-kstack":
        # Qwen2.5-Coder-7B + KStack LoRA fine-tuning at ε=3.  Inherits
        # the trainer's adafactor + BC-off defaults.
        _set("model_name", "Qwen/Qwen2.5-Coder-7B")
        _set("dataset", "JetBrains/KStack")
        _set("dataset_text_field", "content")
        _set("num_train_samples", 50000)
        _set("num_eval_samples", 1000)
        _set("num_epochs", 2)
        _set("batch_size", 192)
        _set("microbatch_size", 16)
        _set("log_steps", 2)
        _set("eval_steps", 10)
        _set("target_epsilon", 3.0)
        _set("learning_rate", 5e-4)
        _set("lora_r", 16)
        _set("lora_alpha", 16)
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
    elif args.preset == "qwen-coder-kstack-lora":
        # Tuned LoRA baseline for Qwen2.5-Coder-7B on KStack, ε=3.
        # Sweep results: r=16 > r=8/24/32/48/64, lr=5e-2 > 2e-2/1e-2,
        # mom=0.9 > 0.8/0.85/0.95/0.99, bs=192 > 128/256/384/512,
        # warmup=0 > 5/10/20.
        # Best eval: 0.3449 at step 520.
        _set("model_name", "Qwen/Qwen2.5-Coder-7B")
        _set("dataset", "JetBrains/KStack")
        _set("dataset_text_field", "content")
        _set("num_train_samples", 50000)
        _set("num_eval_samples", 1000)
        _set("num_epochs", 2)
        _set("batch_size", 192)
        _set("log_steps", 1)
        _set("eval_steps", 10)
        _set("target_epsilon", 3.0)
        _set("learning_rate", 5e-2)
        _set("lora_method", "lora")
        _set("lora_r", 16)
        _set("lora_alpha", 16)
        _set("optimizer", "sgd")
        _set("sgd_momentum", 0.9)
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
    elif args.preset == "roberta-large-glue":
        # The LoRA-XS paper's GLUE setup (arXiv 2405.17604v3, ECAI 2025 --
        # Appendix D.1 and Table 7), reproduced so our rows can sit in their
        # Table 1. Deviating from any of these makes the comparison incomparable,
        # so they are set here rather than left to the caller:
        #   modules  Wq, Wv, Wo, FC1 -- NOT the 7-module set the KStack presets
        #            use. "attention.output.dense" and not "output.dense",
        #            because the latter also matches the FFN's FC2.
        #   alpha    16, fixed across every rank in their sweep
        #   sigma    1e-5 for the R init
        #   seq len  128, batch 32, warmup ratio 0.06
        # AdamW because that is what they used; the per-task learning rate comes
        # from their Table 7 and has to be passed with --learning-rate (it varies
        # by task AND rank, so there is no single defensible default).
        #
        # NOTE eval_steps is per-STEP here, and GLUE tasks are small: RTE is 2.5k
        # examples, so an epoch is ~78 steps at batch 32. 10 would evaluate ~8x
        # per epoch; 25 keeps the eval cost sane while still resolving the curve.
        _set("model_name", "FacebookAI/roberta-large")
        _set("task_type", "sequence-classification")
        _set("num_epochs", 20)
        # 0 means "the whole split". The global defaults are 5000 train / 100
        # eval, sized for a causal-LM smoke test, and both are actively wrong
        # here: 5000 would truncate CoLA (8551), SST-2 (67k) and QNLI (105k),
        # and 100 would cut every validation split down from 277-1043 rows.
        # A correlation over 100 rows is not comparable to anything published.
        _set("num_train_samples", 0)
        _set("num_eval_samples", 0)
        _set("batch_size", 32)
        _set("microbatch_size", 32)
        _set("eval_batch_size", 32)
        _set("log_steps", 1)
        _set("eval_steps", 25)
        _set("learning_rate", 1e-3)
        _set("optimizer", "adamw")
        _set("lora_method", "lora-xs")
        _set("lora_r", 16)
        _set("lora_alpha", 16)
        _set("lora_xs_sigma", 1e-5)
        _set("max_seq_len", 128)
        _set("weight_decay", 0.0)
        _set("lr_schedule", "linear")
        _set("lr_warmup_ratio", 0.06)
        _set(
            "lora_modules",
            ["query", "value", "attention.output.dense", "intermediate.dense"],
        )
        # float32, not bfloat16. The KStack presets run bf16 on a 7B model where
        # memory forces it; RoBERTa-large is 355M, and GLUE's reported figures
        # are correlations (Matthews, Pearson) that are far more sensitive to
        # numerical noise than a token-averaged loss is.
        _set("dtype", "float32")
        # Forced, not defaulted -- see the classification branch below.
        args.attention = "eager"
    elif args.preset == "custom":
        # Keep all user-provided/default CLI arguments unchanged.
        pass

    # --microbatch-size 0 means "no microbatching" (full-batch vmap).
    # Needed because argparse type=int can't accept None on CLI to override presets.
    if args.microbatch_size == 0:
        args.microbatch_size = None

    # --second-moment + --per-group-clipping is not yet supported: the
    # joint first+second-moment privacy allocation has not been validated
    # for PerGroup sensitivities (mirrors the TypeError raised by
    # ``opaque.clipping._clipped_fun``).  Reject early so users see the
    # clear message before either flag is resolved further.
    if args.second_moment != "none" and args.per_group_clipping:
        parser.error(
            "--second-moment is not supported together with "
            "--per-group-clipping: the joint first+second-moment privacy "
            "allocation has not been validated for per-group sensitivities. "
            "Pick one."
        )

    # Parse --per-group-clipping PATTERN=NORM pairs
    if args.per_group_clipping:
        parsed = {}
        fallback_value = None
        for item in args.per_group_clipping:
            if "=" not in item:
                parser.error(
                    f"--per-group-clipping values must be PATTERN=NORM, got '{item}'"
                )
            pattern, value = item.split("=", 1)
            if pattern == "fallback":
                fallback_value = float(value)
            else:
                parsed[pattern] = float(value)
        args.per_group_clipping = parsed
        args.per_group_clipping_fallback = fallback_value

    return args


def main():
    args = parse_args()

    is_ddp, rank, world_size, local_rank = _init_distributed()
    is_main_process = rank == 0

    if args.eval_batch_size is None:
        args.eval_batch_size = args.microbatch_size or args.batch_size

    # Set audit_batch_size to microbatch_size if not specified (forward-only, so at least as cheap)
    if args.audit_batch_size is None:
        args.audit_batch_size = args.microbatch_size or args.batch_size

    print("=" * 80)
    print("DP-SGD LoRA Training for Causal Language Models")
    print("=" * 80)

    # Initialize wandb (enabled by default, offline if no credentials)
    use_wandb = (not args.no_wandb) and is_main_process
    if use_wandb:
        # Generate default run name from key parameters if not specified
        if args.wandb_run_name is None:
            model_short = args.model_name.split("/")[-1]
            run_name = f"{model_short}_n{args.num_train_samples}_e{args.num_epochs}_b{args.batch_size}_eps{args.target_epsilon}_lr{args.learning_rate}"
        else:
            run_name = args.wandb_run_name

        # Offline by default; set WANDB_MODE=online (or WANDB_API_KEY) to sync
        if not os.environ.get("WANDB_MODE"):
            os.environ["WANDB_MODE"] = (
                "online" if os.environ.get("WANDB_API_KEY") else "offline"
            )
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
        )
        print(f"W&B initialized (mode: {os.environ.get('WANDB_MODE', 'online')})")

    # Setup device
    device, device_name = _select_device(local_rank if is_ddp else None)
    if is_main_process:
        if device.type == "cpu":
            print(f"\nUsing device: {device}")
            print("Warning: Training on CPU will be slow")
        else:
            print(f"\nUsing device: {device} ({device_name})")
        if is_ddp:
            print(
                f"Distributed mode: rank={rank}/{world_size}, local_rank={local_rank}"
            )

    # Set seed
    torch.manual_seed(args.seed)

    # Attention implementation: SDPA is the default in recent HuggingFace Transformers
    # and provides up to 3.6x memory savings over eager at seq_len=1024 with vmap.
    # Use --attention eager to override (e.g., for debugging).
    # Classification FORCES eager, regardless of --attention. transformers'
    # SDPA path calls `torch.all(mask == 1)` in
    # _prepare_4d_attention_mask_for_sdpa to skip a no-op mask, and that is
    # data-dependent control flow, which vmap rejects outright. The causal-LM
    # path never trips it only because its collate passes no attention_mask at
    # all -- an encoder has no causal mask to hide padding behind, so it must.
    # Left as a silent default this surfaces as an opaque vmap error several
    # minutes into a GPU run, so it is decided here.
    _cls_forces_eager = args.task_type == "sequence-classification"
    if _cls_forces_eager and args.attention != "eager":
        print(
            "Attention: forcing eager (--attention sdpa is incompatible with "
            "vmap when an attention mask is passed)"
        )
        args.attention = "eager"
    use_eager = args.attention == "eager" or device.type == "mps"

    # When a specific SDPA backend is requested, enable only that one globally.
    if not use_eager and args.sdpa_backend is not None:
        backends = {
            "flash": torch.backends.cuda.enable_flash_sdp,
            "efficient": torch.backends.cuda.enable_mem_efficient_sdp,
            "cudnn": torch.backends.cuda.enable_cudnn_sdp,
            "math": torch.backends.cuda.enable_math_sdp,
        }
        for name, setter in backends.items():
            setter(name == args.sdpa_backend)
        print(f"SDPA backend forced: {args.sdpa_backend}")

    # Resolve the GLUE task before the config, since num_labels feeds into it.
    _is_cls = args.task_type == "sequence-classification"
    glue_task = None
    if _is_cls:
        if not args.glue_task:
            raise ValueError(
                "--task-type sequence-classification requires --glue-task "
                f"(one of: {', '.join(sorted(glue_data.GLUE_TASKS))})"
            )
        glue_task = glue_data.resolve_task(args.glue_task)
        print(
            f"\nGLUE task: {glue_task.name} "
            f"({'regression' if glue_task.is_regression else f'{glue_task.num_labels}-way'}, "
            f"reported metric: {glue_task.metric})"
        )
    elif args.glue_task:
        raise ValueError(
            "--glue-task is only meaningful with "
            "--task-type sequence-classification"
        )
    elif args.classifier_lr is not None:
        raise ValueError(
            "--classifier-lr is only meaningful with "
            "--task-type sequence-classification (there is no head to scale)"
        )

    if _is_cls:
        # Rejected UP FRONT rather than silently ignored. Each of these is
        # meaningless on a classification head, and a run that quietly skips a
        # requested eval looks identical in W&B to one that ran it, which is how
        # ~225 runs came to carry invalid downstream numbers.
        _unsupported = [
            flag
            for flag, on in (
                ("--eval-bpb", getattr(args, "eval_bpb", False)),
                ("--eval-humaneval", getattr(args, "eval_humaneval", False)),
                ("--eval-mbpp", getattr(args, "eval_mbpp", False)),
                ("--audit", getattr(args, "audit", False)),
            )
            if on
        ]
        if args.classifier_lr is not None and args.classifier_lr <= 0:
            raise ValueError(
                f"--classifier-lr must be positive, got {args.classifier_lr}"
            )
        if _unsupported:
            raise SystemExit(
                "these flags are causal-LM only and cannot be used with "
                f"--task-type sequence-classification: {', '.join(_unsupported)}. "
                "BPB is bits-per-byte of a token stream, HumanEval/MBPP are code "
                "generation, and the auditing canaries assume a text corpus. The "
                f"reported figure for {glue_task.name} is eval/{glue_task.metric}."
            )

    # Load model config and disable dropout
    print(f"\nLoading model: {args.model_name}...")
    if _is_cls:
        config = AutoConfig.from_pretrained(
            args.model_name,
            num_labels=glue_task.num_labels,
            # STS-B is a regression task. Without this, HF infers
            # single_label_classification from num_labels=1 and applies
            # cross-entropy to a 1-logit output, which trains to a constant.
            problem_type=(
                "regression" if glue_task.is_regression else "single_label_classification"
            ),
        )
    else:
        config = AutoConfig.from_pretrained(args.model_name)

    dropout_attrs = [
        "attn_pdrop",
        "resid_pdrop",
        "embd_pdrop",
        "attention_dropout",
        "hidden_dropout",
        "dropout",
        "attn_dropout",
        "ffn_dropout",
    ]
    for attr in dropout_attrs:
        if hasattr(config, attr):
            setattr(config, attr, 0.0)

    dtype_name, torch_dtype, dtype_warning = _resolve_model_dtype(args.dtype, device)
    args.dtype = dtype_name
    _print_runtime_mode_report(
        device, device_name, dtype_name, torch_dtype, dtype_warning
    )

    # Load model
    model_kwargs = {
        "config": config,
        "dtype": torch_dtype,
        "trust_remote_code": True,
    }
    if use_eager:
        print("Attention: eager")
        model_kwargs["attn_implementation"] = "eager"
    else:
        backend_label = args.sdpa_backend or "auto"
        print(f"Attention: sdpa (backend={backend_label})")

    # Initialize profiler
    profiler = TrainingProfiler(device)

    _model_cls = (
        AutoModelForSequenceClassification if _is_cls else AutoModelForCausalLM
    )
    try:
        model = _model_cls.from_pretrained(args.model_name, **model_kwargs)
    except TypeError as exc:
        if "unexpected keyword argument 'dtype'" not in str(exc):
            raise
        model_kwargs.pop("dtype")
        model_kwargs["torch_dtype"] = torch_dtype
        model = _model_cls.from_pretrained(args.model_name, **model_kwargs)
    model = model.to(device)
    profiler, _ = profiler.mark("model_loaded")
    print_memory(device, "After model load")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA
    print(f"Applying {args.lora_method}...")
    if args.lora_method == "lora-xs":
        if args.lora_xs_manifold_mode and args.lora_xs_oft_mode:
            raise ValueError(
                "--lora-xs-manifold-mode and --lora-xs-oft-mode are mutually "
                "exclusive — pick one."
            )
        if args.lora_xs_manifold_init_sigma_w and not args.lora_xs_manifold_mode:
            raise ValueError(
                "--lora-xs-manifold-init-sigma-w requires --lora-xs-manifold-mode."
            )
        rank_pattern: dict = {}
        if getattr(args, "lora_xs_rank_pattern_b64", None):
            import base64 as _b64
            import json as _json

            rank_pattern = {
                str(k): int(v)
                for k, v in _json.loads(_b64.b64decode(args.lora_xs_rank_pattern_b64)).items()
            }
            print(
                f"LoRA-XS per-layer rank allocation (inline b64): {len(rank_pattern)} "
                f"overrides; ranks {sorted(set(rank_pattern.values()))}"
            )
        elif getattr(args, "lora_xs_rank_pattern_json", None):
            import json as _json

            with open(args.lora_xs_rank_pattern_json) as _f:
                rank_pattern = {str(k): int(v) for k, v in _json.load(_f).items()}
            print(
                f"LoRA-XS per-layer rank allocation: {len(rank_pattern)} module "
                f"overrides loaded from {args.lora_xs_rank_pattern_json}; "
                f"ranks {sorted(set(rank_pattern.values()))}"
            )
        elif getattr(args, "lora_xs_rank_alloc", "none") == "w0":
            # In-process, data-free per-layer allocation: score each target W0 by
            # its Rényi effective rank (alpha) and allocate the fixed budget
            # sum_l r_l^2 = n_layers * lora_r^2 proportionally. Depends only on the
            # frozen base weights (no data) -> trivially epsilon-free. See
            # docs/renyi-zenml-campaign-plan.md.
            import re as _re

            from lora_privacy.peft_lora_xs.allocation import allocate_from_spectra

            _mods = args.lora_modules if isinstance(args.lora_modules, list) else [args.lora_modules]
            _probe_r = int(getattr(args, "lora_xs_alloc_probe_r", 32))
            _spectra: dict[str, list[float]] = {}
            for _name, _m in model.named_modules():
                _w = getattr(_m, "weight", None)
                if _w is None or _w.ndim != 2:
                    continue
                if not any(_name.endswith(s) or _re.search(rf"(^|\.){s}$", _name) for s in _mods):
                    continue
                _q = min(_probe_r + 8, _w.shape[0], _w.shape[1])
                _sv = torch.linalg.svdvals(_w.detach().float())[:_q] if _q >= min(_w.shape) \
                    else torch.svd_lowrank(_w.detach().float(), q=_q)[1]
                _spectra[_name] = _sv.cpu().tolist()
            _alpha = float(getattr(args, "lora_xs_alloc_alpha", "inf")) if str(
                getattr(args, "lora_xs_alloc_alpha", "inf")).lower() not in ("inf", "infinity") else float("inf")
            rank_pattern = allocate_from_spectra(
                _spectra, mode="renyi", alpha=_alpha, uniform_r=args.lora_r, r_min=2,
            )
            _ach = sum(v * v for v in rank_pattern.values())
            print(
                f"LoRA-XS W0 allocation (alpha={_alpha}): {len(rank_pattern)} layers, "
                f"ranks {sorted(set(rank_pattern.values()))}, budget "
                f"{_ach}/{len(rank_pattern) * args.lora_r ** 2}"
            )
        lora_config = LoraXSConfig(
            r=args.lora_r,
            rank_pattern=rank_pattern,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_modules,
            lora_dropout=0.0,
            sigma=args.lora_xs_sigma,
            orthonormal_a=args.lora_xs_orthonormal_a,
            manifold_mode=args.lora_xs_manifold_mode,
            oft_mode=args.lora_xs_oft_mode,
            manifold_init_sigma_w=args.lora_xs_manifold_init_sigma_w,
            task_type=("SEQ_CLS" if _is_cls else "CAUSAL_LM"),
        )
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_modules,
            lora_dropout=0.0,
            bias="none",
            task_type=("SEQ_CLS" if _is_cls else "CAUSAL_LM"),
            use_dora=args.lora_dora,
            use_rslora=args.lora_rslora,
            init_lora_weights=True if args.lora_init == "default" else args.lora_init,
        )
    model = get_peft_model(model, lora_config)
    apply_model_patches(model)
    model.print_trainable_parameters()

    # Verify per-layer rank allocation actually applied (silent fallback to
    # uniform r would invalidate the whole variable-rank experiment).
    if args.lora_method == "lora-xs" and (
        getattr(args, "lora_xs_rank_pattern_json", None)
        or getattr(args, "lora_xs_rank_pattern_b64", None)
        or getattr(args, "lora_xs_rank_alloc", "none") != "none"
    ):
        _seen = {}
        for _n, _m in model.named_modules():
            _R = getattr(_m, "lora_xs_R", None)
            if _R is not None and "default" in _R:
                _w = getattr(_R["default"], "weight", _R["default"])
                _seen[_n] = int(_w.shape[0])
        _uniq = sorted(set(_seen.values()))
        _n_nonuniform = sum(1 for v in _seen.values() if v != args.lora_r)
        print(f"[rank-alloc] applied ranks {_uniq} across {len(_seen)} LoRA-XS "
              f"layers; {_n_nonuniform} differ from --lora-r={args.lora_r}")
        if _n_nonuniform == 0:
            print("[rank-alloc] WARNING: rank_pattern produced NO per-layer "
                  "variation — check that JSON keys match module names "
                  "(silent fallback to uniform r).")

    profiler, _ = profiler.mark("lora_applied")
    print_memory(device, "After LoRA")

    # Load and prepare dataset
    #
    # Two paths. The causal-LM path streams a text corpus and carves eval off the
    # head of the stream. The GLUE path cannot do that: published GLUE numbers are
    # computed on the task's own `validation` split, so slicing train would make
    # every comparison to the LoRA-XS paper's Table 1 meaningless. glue_data owns
    # that logic and is unit-tested (examples/test_glue_data.py); this branch only
    # wires it in.
    if _is_cls:
        print(f"\nLoading GLUE task: {glue_task.name}")
        try:
            train_dataset, eval_dataset = glue_data.build_glue_datasets(
                glue_task,
                tokenizer,
                max_seq_len=args.max_seq_len,
                # None means "all of it", which is what the paper does. GLUE
                # tasks are small (RTE 2.5k, MRPC 3.7k), so subsampling is
                # opt-in only; the preset sets both counts to 0 = all.
                num_train_samples=(
                    args.num_train_samples if args.num_train_samples > 0 else None
                ),
                num_eval_samples=(
                    args.num_eval_samples if args.num_eval_samples > 0 else None
                ),
                seed=args.seed,
            )
        except ValueError as exc:
            # build_glue_datasets refuses to truncate the validation split.
            # Surfaced as a clean exit, not a traceback: the fix is a flag.
            raise SystemExit(str(exc)) from exc
        if args.num_train_samples > 0:
            print(
                f"  WARNING: subsampling train to {len(train_dataset)} rows; "
                "the paper trains on the full split"
            )
        print(
            f"  train: {len(train_dataset)}  validation: {len(eval_dataset)}  "
            "(GLUE's own validation split)"
        )
        # A 3-tuple, unlike the causal path's 1-tuple. The grad functions below
        # are built with batch_argnums matched to this arity.
        collate = glue_data.make_glue_collate(glue_task, tokenizer, device)
    else:
        # Load and prepare dataset
        print(f"\nLoading dataset: {args.dataset}...")
        if args.dataset_subset:
            print(f"  Subset: {args.dataset_subset}")
        print(f"  Split: {args.dataset_split}")
        print(f"  Text field: {args.dataset_text_field}")

        total_needed = args.num_train_samples + args.num_eval_samples
        dataset = _load_streaming_subset(
            dataset_name=args.dataset,
            dataset_subset=args.dataset_subset,
            dataset_split=args.dataset_split,
            dataset_text_field=args.dataset_text_field,
            total_needed=total_needed,
        )
        print(f"  Total examples in dataset: {len(dataset)}")

        # Validate we have enough data
        if len(dataset) < total_needed:
            raise ValueError(
                f"Dataset has {len(dataset)} examples but need {total_needed} "
                f"(train={args.num_train_samples} + eval={args.num_eval_samples})"
            )

        # Show sample of raw data
        if len(dataset) > 0:
            sample = dataset[0]
            sample_text = sample[args.dataset_text_field]
            print("\n  Sample data (first example):")
            print(f"    Text length: {len(sample_text)} chars")
            print(f"    Preview: {sample_text[:200]}...")

        # Split into eval and train using skip/take
        print(
            f"\nPreparing {args.num_eval_samples} eval + {args.num_train_samples} train samples..."
        )
        eval_dataset = dataset.take(args.num_eval_samples)
        train_dataset = dataset.skip(args.num_eval_samples).take(args.num_train_samples)

        # Tokenize function
        def tokenize_function(examples):
            return tokenizer(
                examples[args.dataset_text_field],
                truncation=True,
                max_length=args.max_seq_len,
            )

        # Tokenize each split separately
        print(f"\nTokenizing (max_seq_len={args.max_seq_len})...")
        eval_cols_to_remove = eval_dataset.column_names
        train_cols_to_remove = train_dataset.column_names

        eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=eval_cols_to_remove,
            desc="Tokenizing eval",
        )
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_cols_to_remove,
            desc="Tokenizing train",
        )

        print(
            f"Prepared datasets: {len(train_dataset)} train samples, {len(eval_dataset)} eval samples"
        )

        # Create data collator (HF primitive for batching + creating labels)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # False for causal LM (GPT-style)
        )

        # Collate: data_collator handles padding, then extract tensors + move to device.
        # Used by all DataLoaders that feed into per_example_loss_fn.
        def collate(examples):
            batch = data_collator(examples)
            return (batch["input_ids"].to(device),)

    # DataLoader batch arity, and therefore which positional args of
    # per_example_loss_fn carry per-example data. Causal passes (input_ids,);
    # classification passes (input_ids, attention_mask, labels). Threading this
    # through as one value keeps the three clipped_grad call sites below from
    # drifting apart -- multi-element batch_argnums is an already-supported path
    # (packages/opaque-core/tests/clipping/test_empty_batch.py).
    _batch_argnums = (1, 2, 3) if _is_cls else (1,)

    # Privacy auditing setup: designate canaries and remove held-out ones
    audit_cf = None
    audit_dataset = None
    audit_ref_scores = None
    if args.audit:
        print(f"\nSetting up privacy auditing with {args.audit_canaries} canaries...")
        audit_cf = auditing.coin_flip(
            train_dataset,
            num_canaries=args.audit_canaries,
            key=key(args.seed),
        )
        audit_dataset = train_dataset  # Keep reference before filtering
        train_dataset = train_dataset.select(audit_cf.train_indices(len(train_dataset)))
        print(
            f"  Canaries: {len(audit_cf.in_indices)} in, "
            f"{len(audit_cf.out_indices)} out (held out from training)"
        )
        print(f"  Training set: {len(train_dataset)} examples")

    global_train_size = len(train_dataset)
    use_shard = is_ddp and args.shard
    use_parallel_poisson = is_ddp and not args.shard
    if use_shard:
        train_dataset = local_shard(train_dataset, rank=rank, world_size=world_size)

    # Eval DataLoader (standard batching, no privacy requirements)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        drop_last=False,
    )

    # For training: Poisson sampling (not uniform shuffling!)
    # Poisson: each example independently sampled with probability sample_rate each step.
    # In parallel Poisson mode each rank samples independently from the full dataset,
    # so we divide by world_size to keep the global expected batch size = args.batch_size.
    use_truncated_poisson = args.sampler == "truncated_poisson"
    max_batch_size = args.max_batch_size or args.batch_size
    sample_rate = args.batch_size / global_train_size
    if use_parallel_poisson:
        sample_rate /= world_size

    expected_steps_per_epoch = int(global_train_size / args.batch_size)

    print("\nPoisson sampling setup:")
    if use_parallel_poisson:
        print(f"  Mode: parallel_poisson (no shard, world_size={world_size})")
    print(f"  Sampler: {args.sampler}")
    if use_truncated_poisson:
        print(f"  Max batch size (cap): {max_batch_size}")
    print(f"  Sample rate (per rank): {sample_rate:.6f}")
    print(f"  Expected global batch size: {args.batch_size}")
    print(f"  Expected steps per epoch: ~{expected_steps_per_epoch}")
    print(f"Eval batches: {len(eval_loader)}")

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
    if args.cpu_offload:
        print(
            f"CPU offload: enabled (save_on_cpu, works {'with' if args.gradient_checkpointing else 'without'} checkpointing)"
        )

    # ---------- LoRA-SB: re-base the frozen factors on the first gradient ----------
    # Placed here deliberately, and it cannot move: it needs the train dataset
    # (built above) and it must precede make_functional below, because that call
    # snapshots the frozen tensors into the functional dict. Re-basing afterwards
    # would change the module while training kept using the old basis.
    if args.lora_method == "lora-xs" and args.lora_xs_init != "weight":
        if args.lora_xs_init == "grad-sb":
            raise SystemExit(
                "--lora-xs-init grad-sb is not available: the gradient probe was "
                "rewritten to return only the top-r factors (it OOM'd holding 196 "
                "full-size targets), and seeding R needs the singular values the "
                "factored probe never materialises. Use --lora-xs-init grad."
            )
        from lora_privacy.peft_lora_xs import rebase_on_gradient

        n_probe = max(1, args.lora_xs_init_batches)
        print(
            f"\nLoRA-SB init: probing {n_probe} batch(es) for the gradient basis "
            f"(mode={args.lora_xs_init})..."
        )
        # batch_size 1: the probe does a full backward through the base model on the
        # plain module path -- no microbatching, no vmap, no functional split. At
        # batch 16 that OOM'd an 80 GB H100. One sequence is enough for a rank-16
        # basis, and LoRA-SB itself probes ~0.1% of the data.
        _probe_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate,
            drop_last=False,
        )
        _probe_batches = list(itertools.islice(_probe_loader, n_probe))

        _pad = tokenizer.pad_token_id

        def _probe_loss(batch):
            # collate() returns a 1-tuple (input_ids,), ALREADY on device -- not a
            # dict. Indexing it with a string raised
            #   TypeError: tuple indices must be integers or slices, not str
            # and killed sb-xse-d5t1-s42 52s in. Same -100 pad masking as
            # per_example_loss_fn, so the probe measures the gradient of the real
            # objective rather than of predicting <eos> after <eos>.
            (ids,) = batch
            labels = ids.masked_fill(ids == _pad, -100) if _pad is not None else ids
            return model(ids, labels=labels).loss

        _t0 = time.time()
        _captured = rebase_on_gradient(
            model,
            _probe_loss,
            _probe_batches,
            args.lora_modules,
            # seed_r bakes LoRA-SB's first step into R as diag(S)*lr/scaling.
            # Default "grad" changes only the BASIS, so a comparison against a
            # W0-basis control moves exactly one thing.
            seed_r=(args.lora_xs_init == "grad-sb"),
            eta=args.learning_rate,
        )
        _vals = sorted(_captured.values())
        print(
            f"  re-based {len(_captured)} layers in {time.time()-_t0:.1f}s; "
            f"captured energy min={_vals[0]:.4f} median={_vals[len(_vals)//2]:.4f} "
            f"max={_vals[-1]:.4f}"
        )
        del _probe_batches, _probe_loader
        for _p in model.parameters():
            _p.grad = None

    # Convert to functional (only LoRA parameters)
    print("\nConverting to functional form (LoRA parameters only)...")
    print("  (This may take 1-2 minutes for large models...)")
    start_time = time.time()
    fmodel, trainable_params, frozen_params = make_functional(
        model,
        disable_autograd_tracking=True,
        partition_trainable=True,
    )
    param_names = list(trainable_params.keys())
    elapsed = time.time() - start_time
    print(f"Trainable parameters: {len(param_names)} (took {elapsed:.1f}s)")
    profiler, _ = profiler.mark("functional_conversion")
    print_memory(device, "After functional conversion")

    def merged_params(trainable):
        return {**frozen_params, **trainable}

    # --- separate learning rate for the classification head ------------------
    # Implemented by scaling the head's UPDATES rather than building a second
    # optimizer, and that is exact rather than an approximation: for SGD, Adam,
    # AdamW and their decoupled weight decay, the update is exactly proportional
    # to the learning rate, so multiplying it by (classifier_lr / learning_rate)
    # yields precisely the update the head would have received at classifier_lr.
    # The second-moment state is computed from gradients and is lr-independent,
    # so sharing it across the two groups changes nothing.
    #
    # It also composes correctly with --lr-schedule: `updates` already carries the
    # scheduled lr, so a constant ratio gives the head the same schedule SHAPE
    # with a different peak, which is what a two-group setup means.
    #
    # PEFT puts the head under modules_to_save, so its keys look like
    # base_model.model.classifier.modules_to_save.default.dense.weight. "score" is
    # the head name some other architectures use.
    _head_keys: list[str] = []
    _head_scale = 1.0
    if _is_cls:
        _head_keys = [
            k for k in trainable_params
            if ".classifier." in f".{k}." or ".score." in f".{k}."
        ]
        if not _head_keys:
            raise SystemExit(
                "could not find the classification head among the trainable "
                "parameters. Without it the head is frozen and the run is "
                f"meaningless. Trainable keys sample: {list(trainable_params)[:3]}"
            )
        if args.classifier_lr is not None:
            _head_scale = args.classifier_lr / args.learning_rate
        print(
            f"Classification head: {len(_head_keys)} tensors, lr="
            f"{args.classifier_lr if args.classifier_lr is not None else args.learning_rate:g}"
            f" (adapter lr={args.learning_rate:g}, update scale={_head_scale:g})"
        )

    # Define per-example loss
    _pad_id = tokenizer.pad_token_id

    def _cls_per_example_loss_fn(trainable, input_ids, attention_mask, labels):
        """Per-example classification loss.

        The batch dim is re-added, deliberately. clipped_grad vmaps over
        batch_argnums, which STRIPS that dim: a (B, L) input arrives here as
        (L,) and a (B,) label as a 0-d scalar. Decoder models tolerate a 1-D
        input_ids, which is why the causal branch below never needs this, but
        RobertaModel.forward unpacks exactly two dims
        (`batch_size, seq_length = input_shape`) and raises otherwise.

        labels.reshape(1) rather than unsqueeze(0) so it also accepts an
        already-1-element tensor, which is what the non-vmap'd probe paths pass.
        """
        output = fmodel(
            merged_params(trainable),
            input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            labels=labels.reshape(1),
        )
        return output.loss

    def _lm_per_example_loss_fn(trainable, input_ids):
        # Mask padding out of the loss.
        #
        # `collate` returns only input_ids and drops the collator's `labels`,
        # which is where DataCollatorForLanguageModeling puts the -100 pad mask.
        # Passing labels=input_ids therefore scored the pad positions, and since
        # pad_token == eos_token that is the model predicting <eos> after <eos>:
        # near-zero loss on ~49.5% of scored positions (mean real length 520 of
        # 1024). It made eval/loss ~half of the true value -- reported 0.3435
        # implies perplexity 1.41 on source code, where the real figure is ~1.98.
        #
        # Reconstructed here rather than threaded through the signature: the
        # vmap'd clipped_grad / auto_clipped_grad / adaptive_clipped_grad call
        # sites bind this as f(trainable, batch_elem), so adding an argument
        # would ripple into the opaque library. This rule is byte-identical to
        # the collator's own (`labels[labels == pad_token_id] = -100`).
        labels = (
            input_ids.masked_fill(input_ids == _pad_id, -100)
            if _pad_id is not None
            else input_ids
        )
        output = fmodel(merged_params(trainable), input_ids, labels=labels)
        return output.loss

    # One name for the rest of the file. The two implementations differ in arity,
    # which is what _batch_argnums above tracks.
    per_example_loss_fn = (
        _cls_per_example_loss_fn if _is_cls else _lm_per_example_loss_fn
    )

    # Build canary DataLoader for auditing
    canary_loader = None
    if args.audit and audit_cf is not None:
        canary_loader = DataLoader(
            audit_cf.canary_subset(audit_dataset),
            batch_size=args.audit_batch_size,
            shuffle=False,
            collate_fn=collate,
        )

    # Auditing helper: compute scores and run one-run estimator
    def run_audit(trainable):
        """Score canaries and report audit metrics. Returns OneRunEstimate or None."""
        if not args.audit or audit_cf is None:
            return None
        scores = auditing.loss_scores(
            per_example_loss_fn,
            trainable,
            batch_argnums=(1,),
            dataloader=canary_loader,
            reference_scores=audit_ref_scores,
        )
        return auditing.one_run(scores, coin_flip=audit_cf)

    # Compute reference (untrained) scores for auditing before any training
    # Paper Algorithm 3: Score = loss(w0, x) - loss(wℓ, x), so we need w0 losses
    if args.audit and audit_cf is not None:
        print("\nComputing reference scores on untrained model...")
        audit_ref_scores = auditing.loss_scores(
            per_example_loss_fn,
            trainable_params,
            batch_argnums=(1,),
            dataloader=canary_loader,
        )
        print(
            f"  Reference scores: mean={audit_ref_scores.mean():.4f}, std={audit_ref_scores.std():.4f}"
        )

    # Set by eval_loss on the classification path: the figure the LoRA-XS paper
    # actually reports for this task (accuracy / Matthews / Pearson). None on the
    # causal path, where eval loss is the reported quantity.
    _last_eval_metric: float | None = None

    def eval_loss(trainable):
        """Mean eval loss over the eval DataLoader.

        Classification runs FULL batches through fmodel directly rather than
        looping per_example_loss_fn: that function re-adds a batch dim for vmap's
        benefit, so feeding it a real (B, L) batch would produce (1, B, L). Full
        batches are also what makes the metric computable -- accuracy, Matthews
        and Pearson all need logits pooled across the whole split, which a scalar
        loss cannot provide. Sets `_last_eval_metric` for the caller to log.
        """
        nonlocal _last_eval_metric
        with torch.no_grad():
            if _is_cls:
                total_loss = 0.0
                total = 0
                logits_all: list[torch.Tensor] = []
                labels_all: list[torch.Tensor] = []
                for input_ids, attention_mask, labels in eval_loader:
                    out = fmodel(
                        merged_params(trainable),
                        input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    total_loss += out.loss.item() * len(input_ids)
                    total += len(input_ids)
                    logits_all.append(out.logits.detach().float())
                    labels_all.append(labels.detach())
                _last_eval_metric = glue_data.glue_metric(
                    glue_task, torch.cat(logits_all), torch.cat(labels_all)
                )
                return total_loss / total

            total_loss = 0.0
            total_tokens = 0
            for (input_ids,) in eval_loader:
                loss = per_example_loss_fn(trainable, input_ids)
                total_loss += loss.item() * len(input_ids)
                total_tokens += len(input_ids)
            return total_loss / total_tokens

    def eval_bpb(trainable, n_samples=512, microbatch=2):
        """Per-example bits-per-byte: token NLL (bits) / UTF-8 bytes of the text.

        Byte-normalised so it is tokenizer-independent and interpretable as
        compression. Returns (aggregate_bpb, [per-example bpb]); the per-example
        list is what enables paired bootstrap / sign-flip tests across arms,
        which a single scalar eval loss cannot support.
        """
        pad_id = tokenizer.pad_token_id
        per_example: list[float] = []
        tot_bits = 0.0
        tot_bytes = 0
        seen = 0
        ln2 = math.log(2.0)
        with torch.no_grad():
            for (input_ids,) in eval_loader:
                if seen >= n_samples:
                    break
                for lo in range(0, input_ids.shape[0], microbatch):
                    if seen >= n_samples:
                        break
                    chunk = input_ids[lo : lo + microbatch]
                    logits = fmodel(merged_params(trainable), chunk).logits
                    # next-token prediction: logits[:, :-1] predicts chunk[:, 1:]
                    logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
                    tgt = chunk[:, 1:]
                    nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # nats
                    valid = (tgt != pad_id) if pad_id is not None else torch.ones_like(tgt, dtype=torch.bool)
                    nll = nll * valid
                    for i in range(chunk.shape[0]):
                        if seen >= n_samples:
                            break
                        ids = chunk[i]
                        if pad_id is not None:
                            ids = ids[ids != pad_id]
                        text = tokenizer.decode(ids, skip_special_tokens=True)
                        nbytes = max(1, len(text.encode("utf-8")))
                        bits = float(nll[i].sum().item()) / ln2
                        per_example.append(bits / nbytes)
                        tot_bits += bits
                        tot_bytes += nbytes
                        seen += 1
                    del logits, logp, nll
        return (tot_bits / max(1, tot_bytes)), per_example

    # Build clipping norm (scalar or per-group)
    if args.per_group_clipping:
        clip_norm = per_group(
            trainable_params,
            fallback=args.per_group_clipping_fallback,
            **args.per_group_clipping,
        )
        if is_main_process:
            print("\nPer-group clipping norms:")
            for gname, val in clip_norm.values.items():
                count = sum(1 for g in clip_norm.groups.values() if g == gname)
                print(f"  {gname}: {val:.3f} ({count} params)")
            print(f"  Effective (for accounting): {clip_norm.effective:.3f}")
    else:
        clip_norm = args.clipping_norm

    # Setup optimizer
    print("\nSetting up DP-SGD training...")
    print(f"  LoRA method: {args.lora_method}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Learning rate: {args.learning_rate}")
    if isinstance(clip_norm, PerGroup):
        print(f"  Clip norm: per-group (effective={clip_norm.effective:.3f})")
    else:
        print(f"  Clip norm: {clip_norm}")
    print(f"  Noise mechanism: {args.noise_mechanism}")
    if args.noise_mechanism != "gaussian":
        print(f"  Noise radius: {args.noise_radius}σ")
    print(f"  Microbatch size: {args.microbatch_size}")
    print(f"  Clipping mode: {args.clipping_mode}")
    if args.clipping_mode == "auto":
        print(f"  AUTO-S gamma: {args.auto_clipping_gamma}")
    print(f"  Eval steps: {args.eval_steps}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Expected total steps: ~{args.num_epochs * expected_steps_per_epoch}")

    # Resolve --second-moment flag to the bool/float overhead value
    # consumed by clipped_grad / acc.second_moment.
    _SECOND_MOMENT_OPTIMIZERS = frozenset(
        {"adam", "adamw", "ademamix", "rmsprop", "radam", "adadelta"}
    )
    second_moment_arg: bool | float
    if args.second_moment == "none":
        second_moment_arg = False
    elif args.second_moment == "auto":
        second_moment_arg = args.optimizer in _SECOND_MOMENT_OPTIMIZERS
    else:
        try:
            second_moment_arg = float(args.second_moment)
        except ValueError as e:
            raise ValueError(
                f"--second-moment must be 'none', 'auto', or a float >1.0, "
                f"got {args.second_moment!r}"
            ) from e
        if second_moment_arg <= 1.0:
            raise ValueError(
                f"--second-moment overhead must be >1.0, got {second_moment_arg}"
            )
    use_second_moment = bool(second_moment_arg)
    if use_second_moment and isinstance(clip_norm, PerGroup):
        raise ValueError(
            "--second-moment is incompatible with --per-group-clipping: the "
            "joint first+second-moment allocation has not been validated for "
            "PerGroup max_norm."
        )

    # Create gradient function based on clipping mode.
    if args.clipping_mode == "adaptive":
        # ``second_moment`` flows through ``**clipped_grad_kwargs`` to the
        # inner ``clipped_grad`` call; the adaptive threshold update reads
        # the first-stream gradient norms regardless of paired-stream output.
        grad_fn, clip_state = adaptive_clipped_grad(
            per_example_loss_fn,
            argnums=0,
            batch_argnums=_batch_argnums,
            initial_clipping_norm=clip_norm,
            target_quantile=1.0 - args.target_clipping_rate,
            clipping_norm_max=args.clipping_norm_max,
            microbatch_size=args.microbatch_size,
            return_aux=True,
            key=key(args.seed),
            normalize_by=args.batch_size,
            second_moment=second_moment_arg,
        )
    elif args.clipping_mode == "auto":
        grad_fn, clip_state = auto_clipped_grad(
            per_example_loss_fn,
            argnums=0,
            batch_argnums=_batch_argnums,
            R=clip_norm,
            gamma=args.auto_clipping_gamma,
            normalize_by=args.batch_size,
            microbatch_size=args.microbatch_size,
            return_aux=True,
            second_moment=second_moment_arg,
        )
    else:
        grad_fn, clip_state = clipped_grad(
            per_example_loss_fn,
            argnums=0,
            batch_argnums=_batch_argnums,
            clipping_norm=clip_norm,
            normalize_by=args.batch_size,
            microbatch_size=args.microbatch_size,
            return_aux=True,
            second_moment=second_moment_arg,
        )

    # Calibrate noise multiplier from target privacy budget
    # sample_rate already computed above
    total_steps = args.num_epochs * expected_steps_per_epoch

    # Compute delta from training set size: δ = 1/n^1.1 (keeps δ below 1/n while
    # being less conservative than the previous 1/n² heuristic on smaller runs).
    if args.target_delta is None:
        args.target_delta = 1.0 / (global_train_size**1.1)
    if use_wandb:
        wandb.config.update({"target_delta": args.target_delta}, allow_val_change=True)

    # Noise injection — bind mechanism-specific parameters once.
    # Chain: base mechanism → adaclip (optional) → amplification.
    # Truncated Gaussian noise provides clipped support but converges to
    # Gaussian for high-dimensional tasks, so we use dpsgd_acc.gaussian() for accounting.
    _num_groups = len(clip_norm.values) if isinstance(clip_norm, PerGroup) else 1
    if args.noise_multiplier == 0:
        mechanism = lambda nm: acc.nonprivate()
    elif args.noise_mechanism == "truncated_gaussian":
        mechanism = dpsgd_acc.gaussian
    else:
        mechanism = dpsgd_acc.gaussian

    if args.clipping_mode == "adaptive":
        _base_mechanism = mechanism
        mechanism = lambda nm, ebs=args.batch_size, ng=_num_groups: dpsgd_acc.adaclip(
            _base_mechanism(nm), expected_batch_size=ebs, num_groups=ng
        )
    if use_second_moment and args.noise_multiplier != 0:
        # second_moment(gaussian(nm)): joint sensitivity = input_sensitivity ·
        # c1 · overhead.  We pass input_sensitivity=1.0 because the runtime
        # expresses noise_stddev as nm · max_norm (mechanism-relative
        # sensitivity is 1).  The accountant scales effective_nm by 1/√(3/2)
        # internally.  Skip the wrap when --noise-multiplier=0 because the
        # underlying mechanism is ``acc.nonprivate()``, which
        # ``acc.second_moment()`` rejects (a non-private mechanism has
        # nothing to add second-moment overhead to).
        _bare_mechanism = mechanism
        _overhead = (
            second_moment_arg if isinstance(second_moment_arg, float) else None
        )
        if _overhead is not None:
            mechanism = lambda nm, oh=_overhead: acc.second_moment(
                _bare_mechanism(nm), sensitivity=1.0, first_moment_overhead=oh,
            )
        else:
            mechanism = lambda nm: acc.second_moment(
                _bare_mechanism(nm), sensitivity=1.0,
            )

    _unamplified = mechanism
    if use_truncated_poisson:
        mechanism = lambda nm: dpsgd_acc.truncated_poisson(
            _unamplified(nm),
            sample_rate=sample_rate,
            batch_size_cap=max_batch_size,
            dataset_size=global_train_size,
        )
    elif use_parallel_poisson:
        mechanism = lambda nm: dpsgd_acc.parallel_poisson(
            _unamplified(nm),
            sample_rate=sample_rate,
            num_workers=world_size,
        )
    else:
        mechanism = lambda nm: dpsgd_acc.poisson(_unamplified(nm), sample_rate=sample_rate)

    # Calibrate noise multiplier from target privacy budget.
    if args.noise_multiplier is not None:
        noise_multiplier = args.noise_multiplier
        print(
            f"\nUsing fixed noise multiplier: {noise_multiplier:.4f} (skipping calibration)"
        )
    else:
        print("\nCalibrating privacy parameters...")
        if use_parallel_poisson:
            print(f"  Accounting: parallel_poisson (world_size={world_size})")
        print(f"  Noise mechanism: {args.noise_mechanism}")
        if args.noise_mechanism == "truncated_gaussian":
            print(f"  Noise radius: {args.noise_radius}σ")
        if use_second_moment:
            print(
                f"  Second-moment release: enabled "
                f"(overhead={second_moment_arg if isinstance(second_moment_arg, float) else 'sqrt(3/2)'})"
            )
        print(f"  δ = {args.target_delta:.2e} (n={global_train_size})")
        print(f"  Total steps: {total_steps}")
        print(f"  Sample rate: {sample_rate:.6f}")
        print(f"  Target: ε={args.target_epsilon}, δ={args.target_delta:.2e}")
        print("  (This may take 1-3 minutes...)")

        start_time = time.time()
        calibration = cal.calibrate(
            cal.epsilon_budget(args.target_epsilon, delta=args.target_delta),
            lambda nm: mechanism(nm) * total_steps,
            param_min=args.calibration_min,
            param_max=args.calibration_max,
            tolerance=args.calibration_tolerance,
        )
        noise_multiplier = calibration.param
        elapsed = time.time() - start_time

        print(f"\nCalibrated privacy parameters (took {elapsed:.1f}s):")
        print(
            f"  Target: ε={args.target_epsilon:.3f}, δ={args.target_delta:.1e} | "
            f"Achieved ε≈{calibration.achieved:.3f}"
        )
        print(
            f"  Noise multiplier: {noise_multiplier:.4f} "
            f"(iterations={calibration.iterations}, converged={calibration.converged})"
        )

    # Build LR schedule using opaque.scheduling primitives.  Each curve
    # returns a ``Callable[[int], float]`` and ``with_warmup`` composes a
    # 0→1 linear ramp during the warmup window; torchopt's
    # ``scale_by_neg_lr`` accepts either a callable or a scalar.  We
    # share ``total_steps`` with the privacy calibration above so the
    # schedule and accounting agree on the run length.  ``--max-steps``
    # (when set) only truncates training — the schedule is laid out over
    # the full planned epoch count, same as accounting.
    if not 0.0 <= args.lr_min_ratio <= 1.0:
        raise ValueError(
            f"--lr-min-ratio must be in [0, 1], got {args.lr_min_ratio}"
        )
    peak_lr = args.learning_rate
    # --lr-warmup-ratio wins when set: GLUE task sizes differ by ~3x, so the
    # paper's 0.06 is only expressible as a fraction of total steps, which is not
    # known until the dataset is loaded.
    if args.lr_warmup_ratio is not None:
        if not 0.0 <= args.lr_warmup_ratio < 1.0:
            raise ValueError(
                f"--lr-warmup-ratio must be in [0, 1), got {args.lr_warmup_ratio}"
            )
        args.lr_warmup_steps = int(round(args.lr_warmup_ratio * total_steps))
        print(
            f"LR warmup: {args.lr_warmup_steps} steps "
            f"({args.lr_warmup_ratio:.3g} x {total_steps} total)"
        )
    warmup = max(0, int(args.lr_warmup_steps))
    lr_min = peak_lr * args.lr_min_ratio
    decay_span = max(1, total_steps - warmup)

    base: float | Schedule
    if args.lr_schedule == "cosine":
        base = cosine_schedule(
            init_value=peak_lr,
            end_value=lr_min,
            transition_steps=decay_span,
            transition_begin=warmup,
        )
    elif args.lr_schedule == "linear":
        base = linear_schedule(
            init_value=peak_lr,
            end_value=lr_min,
            transition_steps=decay_span,
            transition_begin=warmup,
        )
    elif args.lr_schedule == "sqrt":
        # Inverse-sqrt timescale defaults to warmup when set, otherwise
        # to the full training run (gives a gentle ~1/sqrt(2) decay over
        # the run rather than the very aggressive 1/sqrt(t) that would
        # come from a tiny timescale).
        base = inverse_sqrt_schedule(
            init_value=peak_lr,
            transition_steps=warmup if warmup > 0 else max(1, total_steps),
            transition_begin=warmup,
        )
    elif args.lr_schedule == "none":
        base = peak_lr
    else:
        raise ValueError(f"Unknown --lr-schedule: {args.lr_schedule}")

    if warmup > 0:
        lr_for_opt: float | Schedule = with_warmup(base, transition_steps=warmup)
    else:
        lr_for_opt = base

    if args.lr_schedule != "none" or warmup > 0:
        print(
            f"  LR schedule: {args.lr_schedule} "
            f"(peak={peak_lr:g}, min={lr_min:g}, warmup={warmup}, total={total_steps})"
        )

    # Setup optimizer.  Noise metadata travels with ``NoisedPytree`` updates,
    # so optimizer construction does not need a precomputed stddev;
    # ``--noise-bias-correction`` only controls whether the optimizer's
    # DP-aware path consumes that metadata.  For optimizers without a BC
    # path (sgd/lion) the flag is silently ignored.
    if args.optimizer == "adam":
        from opaque.optimizers import adam

        base_opt = adam(
            lr=lr_for_opt,
            weight_decay=args.weight_decay,
            noise_bias_correction=args.noise_bias_correction,
        )
    elif args.optimizer == "sgd":
        _use_xse = (
            args.lora_method == "lora-xs"
            and getattr(args, "lora_xse_p_e", 0.0) > 0
        )
        if _use_xse:
            from lora_privacy.peft_lora_xs import xse_sgd

            base_opt = xse_sgd(
                # lr_for_opt, not args.learning_rate. Every other optimizer branch
                # below takes lr_for_opt (the schedule + warmup wrapper built at
                # ~L1868-1905); this branch took the raw scalar, so --lr-schedule
                # and --lr-warmup-steps were silently inert on the LoRA-XSe path.
                # All 297 runs to date therefore trained at a constant LR.
                # torchopt.sgd takes `lr: ScalarOrSchedule` and _XSeSGD passes it
                # straight through (xse.py:674), so a schedule just works.
                # Default --lr-schedule is still "none", so this changes nothing
                # unless the flag is set: existing runs stay comparable.
                lr=lr_for_opt,
                momentum=args.sgd_momentum,
                p_e=args.lora_xse_p_e,
                lora_alpha=args.lora_alpha,
                rotation_step_interval=args.lora_xse_rotation_step_interval,
                rotation_warmup_steps=args.lora_xse_rotation_warmup_steps,
            )
        else:
            from opaque.optimizers import sgd

            base_opt = sgd(lr=lr_for_opt, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        # LoRA-XSe under AdamW. Until now _use_xse was only reachable from the
        # sgd branch above, so --lora-xse-p-e was silently ignored here and every
        # XSe run in the corpus is heavy-ball SGD. That was an accident of where
        # the check was written, not a property of the method: published LoRA-XS
        # uses AdamW throughout. It also matters, because the one configuration
        # in which full LoRA beats LoRA-XSe on this corpus is the AdamW one.
        _use_xse = (
            args.lora_method == "lora-xs"
            and getattr(args, "lora_xse_p_e", 0.0) > 0
        )
        if _use_xse:
            from lora_privacy.peft_lora_xs import xse_adamw

            base_opt = xse_adamw(
                lr=lr_for_opt,
                lora_alpha=args.lora_alpha,
                p_e=args.lora_xse_p_e,
                rotation_step_interval=args.lora_xse_rotation_step_interval,
                rotation_warmup_steps=args.lora_xse_rotation_warmup_steps,
                # beta2 = 0.99, not the 0.999 default: the second-moment
                # timescale 1/(1-beta2) must not outrun the rotation interval,
                # or freshly inserted directions (whose nu starts at 0) take a
                # first step ~3.2x too large. See xse_adamw's docstring.
                betas=(args.sgd_momentum, args.adam_beta2),
                weight_decay=args.weight_decay,
            )
        else:
            from opaque.optimizers import adamw

            # betas passed explicitly: this path defaulted to (0.9, 0.999) while
            # xse_adamw used (beta1, 0.99), so a frozen-vs-rotating AdamW
            # comparison was silently beta2-mismatched.
            base_opt = adamw(
                lr=lr_for_opt,
                betas=(args.sgd_momentum, args.adam_beta2),
                weight_decay=args.weight_decay,
                noise_bias_correction=args.noise_bias_correction,
            )
    elif args.optimizer == "ademamix":
        from opaque.optimizers import ademamix

        base_opt = ademamix(
            lr=lr_for_opt,
            weight_decay=args.weight_decay,
            noise_bias_correction=args.noise_bias_correction,
        )
    elif args.optimizer == "lion":
        from opaque.optimizers import lion

        base_opt = lion(
            lr=lr_for_opt,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adafactor":
        from opaque.optimizers import adafactor

        base_opt = adafactor(
            lr=lr_for_opt,
            weight_decay=args.weight_decay,
            noise_bias_correction=args.noise_bias_correction,
        )
    elif args.optimizer == "rmsprop":
        from opaque.optimizers import rmsprop

        base_opt = rmsprop(
            lr=lr_for_opt,
            weight_decay=args.weight_decay,
            noise_bias_correction=args.noise_bias_correction,
        )
    elif args.optimizer == "adagrad":
        from opaque.optimizers import adagrad

        base_opt = adagrad(
            lr=lr_for_opt,
            weight_decay=args.weight_decay,
            noise_bias_correction=args.noise_bias_correction,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # Must list every optimizer whose branch above can build a rotation-aware
    # optimizer. This gate controls layer discovery (init needs frozen_params),
    # the frozen= argument to update(), and the rotation/* diagnostics -- so
    # omitting adamw here would construct xse_adamw and then never rotate.
    # Requesting rotation with an optimizer that has no xse wrapper is an ERROR,
    # not a no-op. Only xse_sgd and xse_adamw exist, so with --optimizer adafactor
    # (the trainer's DEFAULT) the branches above build a plain optimizer, this gate
    # is False, and --lora-xse-p-e is silently ignored: the run trains as frozen
    # LoRA-XS while its config says p_e=0.3125 and its name says otherwise. An
    # audit of all 363 runs to date found none affected -- every LoRA-XS run used
    # sgd or adamw -- but nothing was stopping it, and a rotation arm that quietly
    # did not rotate is indistinguishable in W&B from one that did.
    _xse_requested = (
        args.lora_method == "lora-xs" and getattr(args, "lora_xse_p_e", 0.0) > 0
    )
    if _xse_requested and args.optimizer not in ("sgd", "adamw"):
        raise SystemExit(
            f"--lora-xse-p-e {args.lora_xse_p_e} requires --optimizer sgd or "
            f"adamw, got {args.optimizer!r}. Rotation is implemented only as "
            "xse_sgd / xse_adamw; with any other optimizer the rotation would be "
            "silently skipped and the run would secretly be frozen LoRA-XS. "
            "Pass --lora-xse-p-e 0 to train frozen on purpose."
        )
    _xse_active = _xse_requested and args.optimizer in ("sgd", "adamw")
    if _xse_active:
        opt_state = base_opt.init(trainable_params, frozen_params)
    else:
        opt_state = base_opt.init(trainable_params)
    accounting = Accountant()

    # ---------- XSe diagnostic setup ----------
    import re as _re

    _R_KEY_RE_DIAG = _re.compile(
        r"^(?P<prefix>.+)\.lora_xs_R\.(?P<adapter>[^.]+)\.weight$"
    )
    _xse_diag_layers: list[dict] = []
    _xse_p_e = getattr(args, "lora_xse_p_e", 0.0)
    if args.lora_method == "lora-xs":
        import math as _math
        import optree as _optree

        _diag_leaves, _diag_treedef = _optree.tree_flatten(trainable_params)
        _diag_flat_idx = _optree.tree_unflatten(
            _diag_treedef, list(range(len(_diag_leaves)))
        )
        assert isinstance(_diag_flat_idx, dict)
        for _r_key in trainable_params.keys():
            _m = _R_KEY_RE_DIAG.match(_r_key)
            if _m is None:
                continue
            _prefix = _m.group("prefix")
            _r = trainable_params[_r_key].shape[0]
            _r_e = int(_math.floor(_xse_p_e * _r))
            _xse_diag_layers.append(
                {
                    "prefix": _prefix,
                    "r_key": _r_key,
                    "flat_index": _diag_flat_idx[_r_key],
                    "r": _r,
                    "r_e": _r_e,
                    "r_keep": _r - _r_e,
                }
            )

    # Per-step tracking for XSe continuous metrics.
    _prev_r_explore_norm: dict[str, float] = {}  # r_key → previous ‖R[r_keep:,:]‖

    # Ring buffer for trailing loss slope.
    _loss_ring: list[float] = []
    _LOSS_SLOPE_WINDOW = 20

    # Noise functions consume ClippedPytree metadata directly and return
    # NoisedPytree updates carrying the realized per-step stddev.
    initial_bound = clip_norm / args.batch_size
    if args.noise_mechanism == "truncated_gaussian" and noise_multiplier != 0:
        noise_fn, noise_state = truncated_gaussian_noise(
            noise_multiplier=noise_multiplier,
            radius=args.noise_radius,
            key=key(args.seed),
        )
    else:
        noise_fn, noise_state = gaussian_noise(
            noise_multiplier=noise_multiplier,
            key=key(args.seed),
        )

    # Training loop
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    losses = []
    clip_norms_history = []
    clip_rates_history = []
    last_clip_bound = initial_bound
    global_step = 0

    # Reset peak memory before training to get accurate training peak
    reset_peak_memory(device)
    profiler, _ = profiler.mark("training_start")
    print_memory(device, "Before training")

    # Step-0 eval: log baseline metrics before any training
    initial_eval_loss = eval_loss(trainable_params)
    initial_epsilon = accounting.epsilon_at(args.target_delta)
    initial_noise_std = _noise_stddev(initial_bound, noise_multiplier)
    print(f"  → Step 0 eval: loss={initial_eval_loss:.4f}, ε={initial_epsilon:.3f}")
    if use_wandb:
        wandb.log(
            {
                "eval/loss": initial_eval_loss,
                "privacy/epsilon": initial_epsilon,
                "train/noise_std": _effective(initial_noise_std),
                "train/clipping_norm": _effective(clip_norm),
            },
            step=0,
        )

    # --- Eval-noise bookkeeping (best-checkpoint + EMA) ---
    # best_eval_loss / best_eval_step track the lowest eval/loss seen.
    # best_snapshot holds a CPU copy of the trainable params at that step; it is
    # restored before saving + downstream eval when --restore-best-checkpoint.
    # eval_loss_ema is a smoothed metric robust to per-checkpoint eval noise.
    best_eval_loss = initial_eval_loss
    best_eval_step = 0
    # GLUE's reported figures are higher-is-better, so they need their own
    # running max rather than reusing the loss minimum. Seeded from the
    # pre-training eval so a run that never improves still reports a number.
    best_glue_score: float | None = _last_eval_metric
    best_glue_step = 0
    best_snapshot = (
        {k: v.detach().cpu().clone() for k, v in trainable_params.items()}
        if args.restore_best_checkpoint and is_main_process
        else None
    )
    eval_loss_ema = initial_eval_loss
    _ema_beta = float(args.eval_ema_beta)

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 80)
        print(f"Creating {args.sampler} sampler...")

        # Create sampler for this epoch
        if use_truncated_poisson:
            epoch_sampler = TruncatedPoissonSampler(
                train_dataset,
                sample_rate=sample_rate,
                max_batch_size=max_batch_size,
                num_iterations=expected_steps_per_epoch,
                key=fold_in(key(args.seed), rank, epoch),
            )
        else:
            epoch_sampler = PoissonSampler(
                train_dataset,
                sample_rate=sample_rate,
                num_iterations=expected_steps_per_epoch,
                key=fold_in(key(args.seed), rank, epoch),
            )
        print("Creating DataLoader...")

        # DataLoader with batch_sampler
        epoch_loader = DataLoader(
            train_dataset,
            batch_sampler=epoch_sampler,
            collate_fn=collate,
        )

        # Iterate through Poisson-sampled batches
        for step_idx, batch in enumerate(epoch_loader):
            # batch is (input_ids,) on the causal path and
            # (input_ids, attention_mask, labels) on the classification one;
            # element 0 is input_ids either way, which is all this loop needs
            # directly. The rest is splatted into grad_fn below.
            input_ids = batch[0]

            # === Accounting (data-independent, before execution) ===
            accounting |= mechanism(noise_multiplier)

            batch_size = len(input_ids)

            # === Execution ===
            step_timer = StepTimer(device, batch_size=batch_size)
            with step_timer:
                # Compute clipped gradients (handles empty batches via library)
                with offload_ctx:
                    (grads_tuple, aux), clip_state = grad_fn(
                        trainable_params, *batch, state=clip_state
                    )
                if is_ddp:
                    clip_state, aux = sync(clip_state, aux)
                    sum_gradients_(grads_tuple)

                step_clip_norm = _step_clip_norm(grads_tuple)
                noisy_grads, noise_state = noise_fn(grads_tuple, noise_state)
                noise_stddev = _step_noise_stddev(noisy_grads)
                if is_ddp:
                    noise_state = sync(noise_state)

                if _xse_active:
                    updates, opt_state, frozen_params = base_opt.update(
                        noisy_grads,
                        opt_state,
                        params=trainable_params,
                        frozen=frozen_params,
                    )
                else:
                    updates, opt_state = base_opt.update(
                        noisy_grads, opt_state, params=trainable_params
                    )
                if _head_scale != 1.0:
                    updates = dict(updates)
                    for _hk in _head_keys:
                        _hu = updates.get(_hk)
                        if _hu is not None:
                            updates[_hk] = _hu * _head_scale
                trainable_params = torchopt.apply_updates(trainable_params, updates)

            profiler = profiler.add_step(step_timer)

            # Empty batch (rare but possible with Poisson): skip metrics.
            if batch_size == 0:
                global_step += 1
                continue

            # === Step metrics ===
            avg_loss = aux.loss_values.mean().item()
            clip_rate = aux.clipping_rate
            mean_grad_norm = aux.grad_norms.mean().item()

            losses.append(avg_loss)
            clip_norms_history.append(_effective(step_clip_norm))
            clip_rates_history.append(clip_rate)
            last_clip_bound = step_clip_norm

            # Trailing loss ring for slope-of-loss metric (see wandb block below).
            _loss_ring.append(avg_loss)
            if len(_loss_ring) > _LOSS_SLOPE_WINDOW:
                _loss_ring.pop(0)

            global_step += 1

            # === Logging (every log_steps) ===
            if global_step % args.log_steps == 0:
                log_profiler = sync(profiler) if is_ddp else profiler
                profiler = log_profiler
                perf = profiler.current_metrics()

                if use_wandb:
                    wb_metrics = {
                        "train/loss": avg_loss,
                        "train/batch_size": batch_size,
                        "train/clipping_norm": _effective(step_clip_norm),
                        "train/clip_rate": clip_rate,
                        "train/grad_norm_mean": mean_grad_norm,
                        "train/clipped_grad_norm_mean": aux.clipped_grad_norms.mean().item(),
                        "train/noise_std": _effective(noise_stddev),
                        "perf/step_time_sec": perf["step_time_sec"],
                        "perf/throughput_samples_per_sec": perf[
                            "throughput_samples_sec"
                        ],
                        "perf/allocated_gb": perf["memory_allocated_gb"],
                        "perf/reserved_gb": perf["memory_reserved_gb"],
                        "perf/peak_gb": perf["memory_peak_gb"],
                    }
                    # ---- Trailing loss slope ----
                    if len(_loss_ring) >= 2:
                        _slope_win = _loss_ring[-_LOSS_SLOPE_WINDOW:]
                        _slope = (_slope_win[-1] - _slope_win[0]) / max(
                            len(_slope_win) - 1, 1
                        )
                        wb_metrics["train/loss_slope"] = _slope

                    # =============================================================
                    # LoRA-XSe diagnostics
                    # =============================================================
                    if _xse_diag_layers:
                        from lora_privacy.core.svd import _spectral_entropy, _svdvals

                        # -- Per-layer aggregation --
                        _r_frob_sum = 0.0
                        _r_info_sum = 0.0
                        _m_frob_sum = 0.0
                        _m_info_sum = 0.0
                        _r_keep_norm_sum = 0.0
                        _r_explore_norm_sum = 0.0
                        _m_keep_norm_sum = 0.0
                        _m_explore_norm_sum = 0.0
                        _r_explore_info_sum = 0.0
                        _grad_explore_frac_sum = 0.0
                        _grad_snr_sum = 0.0
                        _m_explore_ratio_sum = 0.0
                        _r_explore_growth_sum = 0.0
                        _r_velocity_sum = 0.0
                        _r_condition_sum = 0.0
                        _r_block_coherence_sum = 0.0
                        _n_layers = 0
                        _n_explore_growth = 0
                        # Per-layer momentum-spectrum collection (rank-reallocation
                        # probe): does recoverable rank vary ACROSS layers?
                        _pl_m_info = []     # per-layer M_R normalized Shannon entropy
                        _pl_rec_rank = []   # per-layer #singular values > 2*median (spike count)
                        _pl_top_ratio = []  # per-layer s_max / median (top-spike strength)

                        _inner_state_for_diag = (
                            opt_state.inner
                            if hasattr(opt_state, "inner")
                            else opt_state
                        )
                        _has_trace = (
                            _inner_state_for_diag is not None
                            and hasattr(_inner_state_for_diag[0], "trace")
                            and _inner_state_for_diag[0].trace
                        )

                        for _li in _xse_diag_layers:
                            _R = trainable_params[_li["r_key"]]
                            _R_f = _R.detach().to(torch.float32)
                            _r_svs = _svdvals(_R_f)
                            _r_frob_sum += float(torch.linalg.norm(_R_f).item())
                            _r_info_sum += _spectral_entropy(_r_svs)
                            _n_layers += 1

                            _r_keep = _li["r_keep"]
                            _r_e = _li["r_e"]

                            # R condition number: σ_max / σ_min
                            _s_max = float(_r_svs[0].item())
                            _s_min = float(_r_svs[-1].clamp(min=1e-12).item())
                            _r_condition_sum += _s_max / _s_min

                            # R velocity: ‖update‖ (how fast R is changing)
                            _upd = updates.get(_li["r_key"])
                            if _upd is not None:
                                _r_velocity_sum += float(
                                    torch.linalg.norm(_upd.detach().to(torch.float32)).item()
                                )

                            if _r_e > 0:
                                _r_keep_norm_f = float(
                                    torch.linalg.norm(_R_f[:_r_keep, :_r_keep]).item()
                                )
                                _r_explore_norm_f = float(
                                    torch.linalg.norm(_R_f[_r_keep:, :]).item()
                                )
                                _r_keep_norm_sum += _r_keep_norm_f
                                _r_explore_norm_sum += _r_explore_norm_f
                                _explore_svs = _svdvals(_R_f[_r_keep:, _r_keep:])
                                _r_explore_info_sum += _spectral_entropy(_explore_svs)

                                # R block coherence: ‖R_kk‖ / ‖R‖
                                _r_full_norm = float(torch.linalg.norm(_R_f).item())
                                if _r_full_norm > 1e-12:
                                    _r_block_coherence_sum += _r_keep_norm_f / _r_full_norm

                                # R explore growth: Δ‖R[r_keep:,:]‖
                                _rk = _li["r_key"]
                                if _rk in _prev_r_explore_norm:
                                    _r_explore_growth_sum += (
                                        _r_explore_norm_f - _prev_r_explore_norm[_rk]
                                    )
                                    _n_explore_growth += 1
                                _prev_r_explore_norm[_rk] = _r_explore_norm_f

                            # Gradient-based metrics (from noisy DP gradient)
                            # noisy_grads is a NoisedPytree wrapper; the actual
                            # dict of named tensors lives in .pytree.
                            _grads_dict = (
                                noisy_grads.pytree if hasattr(noisy_grads, "pytree")
                                else noisy_grads
                            )
                            _g = _grads_dict.get(_li["r_key"])
                            if _g is not None:
                                _g_f = _g.detach().to(torch.float32)
                                _g_norm = float(torch.linalg.norm(_g_f).item())

                                # grad_explore_frac: ‖g[r_keep:,:]‖ / ‖g‖
                                if _r_e > 0 and _g_norm > 1e-12:
                                    _g_explore = float(
                                        torch.linalg.norm(_g_f[_r_keep:, :]).item()
                                    )
                                    _grad_explore_frac_sum += _g_explore / _g_norm

                            if _has_trace:
                                _m_R = _inner_state_for_diag[0].trace[
                                    _li["flat_index"]
                                ].to(torch.float32)
                                _m_norm_f = float(torch.linalg.norm(_m_R).item())
                                _m_frob_sum += _m_norm_f
                                _m_svs = _svdvals(_m_R)
                                _m_ent = _spectral_entropy(_m_svs)
                                _m_info_sum += _m_ent
                                # Per-layer collection for the reallocation probe.
                                _m_med = float(_m_svs.median().clamp(min=1e-12).item())
                                _pl_m_info.append(float(_m_ent))
                                _pl_rec_rank.append(int((_m_svs > 2.0 * _m_med).sum().item()))
                                _pl_top_ratio.append(float((_m_svs[0] / _m_med).item()))

                                # Continuous m_explore_ratio: ‖m[r_keep:,:]‖/‖m[:r_keep,:]‖
                                if _r_e > 0:
                                    _m_keep_n = float(
                                        torch.linalg.norm(_m_R[:_r_keep, :]).item()
                                    )
                                    _m_explore_n = float(
                                        torch.linalg.norm(_m_R[_r_keep:, :]).item()
                                    )
                                    _m_keep_norm_sum += _m_keep_n
                                    _m_explore_norm_sum += _m_explore_n
                                    _m_explore_ratio_sum += (
                                        _m_explore_n / max(_m_keep_n, 1e-12)
                                    )

                                # grad signal-to-noise: ‖m_R‖ / ‖g_R - m_R‖
                                if _g is not None:
                                    _g_f = _g.detach().to(torch.float32)
                                    _residual = float(
                                        torch.linalg.norm(_g_f - _m_R).item()
                                    )
                                    if _residual > 1e-12:
                                        _grad_snr_sum += _m_norm_f / _residual

                        # -- Continuous metrics (xs/) --
                        wb_metrics["xs/r_norm"] = _r_frob_sum / _n_layers
                        wb_metrics["xs/r_info"] = _r_info_sum / _n_layers
                        wb_metrics["xs/r_condition"] = _r_condition_sum / _n_layers
                        wb_metrics["xs/r_velocity"] = _r_velocity_sum / _n_layers
                        if _has_trace:
                            wb_metrics["xs/m_norm"] = _m_frob_sum / _n_layers
                            wb_metrics["xs/m_info"] = _m_info_sum / _n_layers
                            wb_metrics["xs/grad_snr"] = _grad_snr_sum / _n_layers
                        if _xse_p_e > 0 and _n_layers > 0:
                            wb_metrics["xs/r_keep_norm"] = _r_keep_norm_sum / _n_layers
                            wb_metrics["xs/r_explore_norm"] = _r_explore_norm_sum / _n_layers
                            wb_metrics["xs/r_explore_info"] = _r_explore_info_sum / _n_layers
                            wb_metrics["xs/r_block_coherence"] = _r_block_coherence_sum / _n_layers
                            wb_metrics["xs/grad_explore_frac"] = _grad_explore_frac_sum / _n_layers
                            if _has_trace:
                                wb_metrics["xs/m_keep_norm"] = _m_keep_norm_sum / _n_layers
                                wb_metrics["xs/m_explore_norm"] = _m_explore_norm_sum / _n_layers
                                wb_metrics["xs/m_explore_ratio"] = _m_explore_ratio_sum / _n_layers
                            if _n_explore_growth > 0:
                                wb_metrics["xs/r_explore_growth"] = (
                                    _r_explore_growth_sum / _n_explore_growth
                                )
                        # Effective rank: exp(H * log(r)) where H is spectral entropy.
                        if _n_layers > 0 and _xse_diag_layers:
                            import math as _m2
                            _avg_info = _r_info_sum / _n_layers
                            _avg_r = sum(li["r"] for li in _xse_diag_layers) / _n_layers
                            wb_metrics["xs/r_effective_rank"] = _m2.exp(
                                _avg_info * _m2.log(max(_avg_r, 1))
                            )

                        # -- Per-layer M_R spread (rank-reallocation probe) --
                        # The decisive question: do layers DIFFER in recoverable
                        # rank? If rec_rank_max >> rec_rank_min, reallocation has
                        # headroom; if all layers look alike, the idea is dead.
                        if _pl_m_info:
                            def _spread(xs):
                                s = sorted(xs); n = len(s); mu = sum(xs) / n
                                q = lambda p: s[min(n - 1, int(p * (n - 1) + 0.5))]
                                return (s[0], q(0.5), s[-1],
                                        (sum((x - mu) ** 2 for x in xs) / n) ** 0.5)
                            _mi = _spread(_pl_m_info)
                            wb_metrics["xs_spread/m_info_min"] = _mi[0]
                            wb_metrics["xs_spread/m_info_median"] = _mi[1]
                            wb_metrics["xs_spread/m_info_max"] = _mi[2]
                            wb_metrics["xs_spread/m_info_std"] = _mi[3]
                            wb_metrics["xs_spread/m_info_hist"] = wandb.Histogram(_pl_m_info)
                            _rr = _spread([float(x) for x in _pl_rec_rank])
                            wb_metrics["xs_spread/rec_rank_min"] = _rr[0]
                            wb_metrics["xs_spread/rec_rank_median"] = _rr[1]
                            wb_metrics["xs_spread/rec_rank_max"] = _rr[2]
                            wb_metrics["xs_spread/rec_rank_std"] = _rr[3]
                            wb_metrics["xs_spread/rec_rank_hist"] = wandb.Histogram(_pl_rec_rank)
                            wb_metrics["xs_spread/top_ratio_median"] = _spread(_pl_top_ratio)[1]
                            wb_metrics["xs_spread/n_layers"] = len(_pl_m_info)

                        # -- Rotation-event metrics (rotation/) --
                        if _xse_active and getattr(opt_state, "last_diag", None):
                            diag = opt_state.last_diag
                            per_layer = diag.get("per_layer", {})
                            if diag.get("rotated", False):
                                _mean = lambda xs: sum(xs) / len(xs)
                                def _agg(key):
                                    vals = [e[key] for e in per_layer.values() if key in e]
                                    return _mean(vals) if vals else None

                                _v = _agg("r_norm_old")
                                _vn = _agg("r_norm_new")
                                if _v and _v > 1e-12:
                                    # RATIO OF MEANS -- kept for continuity with
                                    # every run to date, but it is dominated by the
                                    # few layers with the largest ||R||. Simulated:
                                    # per-layer ratios averaging 0.925 log as 0.743
                                    # when 20 large-||R|| layers retain 0.70 and 180
                                    # small ones retain 0.95.
                                    wb_metrics["rotation/r_norm_growth"] = _vn / _v
                                # MEAN OF RATIOS -- the quantity the exact identity
                                # ||dW' - dW||/||dW|| = sqrt(1 - g^2) is about, since
                                # that identity is per-layer. Read this one.
                                _pl = [
                                    e["r_norm_new"] / e["r_norm_old"]
                                    for e in per_layer.values()
                                    if e.get("r_norm_old", 0.0) > 1e-12
                                    and "r_norm_new" in e
                                ]
                                if _pl:
                                    wb_metrics["rotation/r_norm_growth_perlayer"] = (
                                        sum(_pl) / len(_pl)
                                    )
                                    wb_metrics["rotation/r_norm_growth_min"] = min(_pl)
                                    wb_metrics["rotation/r_norm_growth_max"] = max(_pl)
                                _v = _agg("subspace_sin")
                                if _v is not None:
                                    wb_metrics["rotation/r_subspace_angle"] = _v
                                _v = _agg("m_norm_old")
                                _vn = _agg("m_norm_new")
                                if _v and _v > 1e-12:
                                    wb_metrics["rotation/m_norm_growth"] = _vn / _v
                                _v = _agg("spectral_gap")
                                if _v is not None:
                                    wb_metrics["rotation/spectral_gap"] = _v
                                _v = _agg("projection_energy")
                                if _v is not None:
                                    wb_metrics["rotation/projection_energy"] = _v
                                _v = _agg("explore_m_ratio")
                                if _v is not None:
                                    wb_metrics["rotation/explore_m_ratio"] = _v
                                _v = _agg("promotion_count")
                                if _v is not None:
                                    wb_metrics["rotation/promotion_count"] = _v
                                _v = _agg("energy_ratio")
                                if _v is not None:
                                    wb_metrics["rotation/energy_ratio"] = _v
                                _v = _agg("r_cross_norm")
                                if _v is not None:
                                    wb_metrics["rotation/r_cross_norm"] = _v
                                # Momentum spectrum, layer-mean of each index.
                                # sv0..sv7 make the signal/noise cut point
                                # locatable: pure iid noise of per-entry std s
                                # fills [0, 2*s*sqrt(r)], so the bulk edge can be
                                # read off the tail instead of assumed.
                                for _i in range(8):
                                    _v = _agg(f"sv{_i}")
                                    if _v is not None:
                                        wb_metrics[f"rotation/sv{_i}"] = _v
                                # min/max of r_e ACROSS layers. The mean alone
                                # hides clamping (r_e clipped to 1 when
                                # floor(N_alpha)+margin >= r); r_e_min == 1 with a
                                # mean well above 1 is the clamping signature.
                                _re = [e["r_e_layer"] for e in per_layer.values()
                                       if "r_e_layer" in e]
                                if _re:
                                    wb_metrics["rotation/r_e_min"] = min(_re)
                                    wb_metrics["rotation/r_e_max"] = max(_re)
                                    wb_metrics["rotation/r_e_frac_clamped"] = (
                                        sum(1 for _x in _re if _x <= 1.0) / len(_re)
                                    )
                                # Rényi entropy diagnostics (alpha set by
                                # XSE_ADAPTIVE_DEPTH_ALPHA env var; at α=1
                                # these equal the Shannon-based metrics).
                                _v = _agg("spectral_entropy_renyi")
                                if _v is not None:
                                    wb_metrics["rotation/spectral_entropy_renyi"] = _v
                                _v = _agg("r_eff_renyi")
                                if _v is not None:
                                    wb_metrics["rotation/r_eff_renyi"] = _v
                                _v = _agg("r_e_dyn")
                                if _v is not None:
                                    wb_metrics["rotation/r_e_dyn"] = _v
                                _v = _agg("alpha")
                                if _v is not None:
                                    wb_metrics["rotation/alpha"] = _v
                                # Diagnostic Rényi effective-rank α-grid, mean
                                # over layers. The DP-inflation test: under DP
                                # the low-α values (a0/a0p5) sit far above the
                                # stable rank (ainf); under non-DP the curve is
                                # flatter. See docs/renyi-effective-rank-theory.md.
                                for _gk in ("r_eff_a0", "r_eff_a0p5", "r_eff_a1",
                                            "r_eff_a2", "r_eff_ainf"):
                                    _v = _agg(_gk)
                                    if _v is not None:
                                        wb_metrics[f"rotation/{_gk}"] = _v
                                # Noise-inflation gap: low-α minus stable rank.
                                _lo = _agg("r_eff_a0p5")
                                _hi = _agg("r_eff_ainf")
                                if _lo is not None and _hi is not None:
                                    wb_metrics["rotation/renyi_gap_a0p5_ainf"] = _lo - _hi
                    # Per-group metrics under group/ section
                    if (
                        isinstance(step_clip_norm, PerGroup)
                        and aux.group_norms is not None
                    ):
                        for gname in step_clip_norm.values:
                            gn_bound = step_clip_norm.values[gname]
                            wb_metrics[f"group/clipping_norm/{gname}"] = gn_bound
                            gnorms = aux.group_norms[gname]
                            wb_metrics[f"group/grad_norm/{gname}"] = (
                                gnorms.mean().item()
                            )
                            gn_clipped = float((gnorms > gn_bound).sum().item())
                            wb_metrics[f"group/clip_rate/{gname}"] = gn_clipped / max(
                                1.0, float(batch_size)
                            )
                    wandb.log(wb_metrics, step=global_step)

                print(
                    f"Step {global_step:4d} [E{epoch + 1} S{step_idx + 1:3d}/{expected_steps_per_epoch:3d}] | "
                    f"BS: {batch_size} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Clip: norm={_effective(step_clip_norm):.3f}, rate={clip_rate:.1%} | "
                    f"GradNorm: μ={mean_grad_norm:.3f} | "
                    f"Noise: σ={_effective(noise_stddev):.4f} | "
                    f"Time: {perf['step_time_sec']:.2f}s | Mem: {perf['memory_peak_gb']:.1f}GB"
                )

            # Expensive operations (eval + privacy + audit) every eval_steps
            if global_step % args.eval_steps == 0:
                current_eval_loss = eval_loss(trainable_params)
                # Cache PLD before eval so it serves as opaque boundary
                accounting = acc.cached(accounting)
                epsilon = accounting.epsilon_at(args.target_delta)

                # Track best checkpoint + EMA (eval-noise fix).
                if _ema_beta > 0:
                    eval_loss_ema = (
                        _ema_beta * eval_loss_ema + (1.0 - _ema_beta) * current_eval_loss
                    )
                if current_eval_loss < best_eval_loss:
                    best_eval_loss = current_eval_loss
                    best_eval_step = global_step
                    if best_snapshot is not None:
                        for k, v in trainable_params.items():
                            best_snapshot[k].copy_(v.detach().to("cpu"))

                metrics = {
                    "eval/loss": current_eval_loss,
                    "eval/loss_min": best_eval_loss,
                    "privacy/epsilon": epsilon,
                }
                if _ema_beta > 0:
                    metrics["eval/loss_ema"] = eval_loss_ema
                eval_msg = (
                    f"  → Eval: loss={current_eval_loss:.4f} "
                    f"(min={best_eval_loss:.4f}@{best_eval_step}), ε={epsilon:.3f}"
                )
                # On GLUE the loss is not the reported quantity -- the paper's
                # Table 1 is accuracy / Matthews / Pearson. Logged under its own
                # task-specific key AND a generic one, so a sweep across tasks can
                # be aggregated without knowing which metric each task uses.
                # `_best` tracks the max because all three are higher-is-better,
                # unlike loss.
                if _is_cls and _last_eval_metric is not None:
                    metrics[f"eval/{glue_task.metric}"] = _last_eval_metric
                    metrics["eval/glue_score"] = _last_eval_metric
                    if (
                        best_glue_score is None
                        or _last_eval_metric > best_glue_score
                    ):
                        best_glue_score = _last_eval_metric
                        best_glue_step = global_step
                    metrics["eval/glue_score_max"] = best_glue_score
                    eval_msg += (
                        f", {glue_task.metric}={_last_eval_metric:.4f} "
                        f"(max={best_glue_score:.4f}@{best_glue_step})"
                    )

                if args.audit:
                    audit_estimate = run_audit(trainable_params)
                    audit_eps = audit_estimate.epsilon_at(delta=args.target_delta)
                    audit_auc = audit_estimate.auc()
                    metrics["privacy/epsilon_empirical"] = audit_eps
                    metrics["privacy/audit_auc"] = audit_auc
                    eval_msg += f", ε_audit={audit_eps:.4f}, AUC={audit_auc:.4f}"

                if use_wandb:
                    wandb.log(metrics, step=global_step)
                print(eval_msg)

            # Early exit if max_steps reached
            if args.max_steps is not None and global_step >= args.max_steps:
                print(f"\nReached max_steps={args.max_steps}, stopping training.")
                break

        # Break outer epoch loop if max_steps reached
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    # Final summary
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset} ({global_train_size} train samples)")
    print("\nTraining results:")
    print(f"  Total steps: {global_step}")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")

    if args.clipping_mode == "adaptive":
        print("\nAdaptive clipping:")
        if isinstance(last_clip_bound, PerGroup):
            print("  Per-group output bounds:")
            initial_cn = initial_bound
            for gname in sorted(last_clip_bound.values.keys()):
                print(
                    f"    {gname}: {initial_cn.values[gname]:.3f} → {last_clip_bound.values[gname]:.3f}"
                )
            print(f"  Effective (final): {last_clip_bound.effective:.3f}")
        else:
            print(f"  Initial output max_norm: {_effective(initial_bound):.3f}")
            print(f"  Final output max_norm: {last_clip_bound:.3f}")
        print(
            f"  Clip norm range: [{min(clip_norms_history):.3f}, {max(clip_norms_history):.3f}]"
        )
    elif isinstance(clip_norm, PerGroup):
        print("\nPer-group clipping:")
        for gname, val in clip_norm.values.items():
            print(f"  {gname}: {val:.3f}")
        print(f"  Effective: {clip_norm.effective:.3f}")
        print(
            f"  Average clip rate: {sum(clip_rates_history) / len(clip_rates_history):.2%}"
        )
    else:
        print("\nFixed clipping:")
        print(f"  Clip norm: {args.clipping_norm:.3f}")
        print(
            f"  Average clip rate: {sum(clip_rates_history) / len(clip_rates_history):.2%}"
        )

    final_epsilon = accounting.epsilon_at(args.target_delta)
    print("\nPrivacy:")
    if use_truncated_poisson:
        print(
            f"  Accounting: truncated_poisson (cap={max_batch_size}, n={global_train_size})"
        )
    elif use_parallel_poisson:
        print(f"  Accounting: parallel_poisson (world_size={world_size})")
    if use_second_moment:
        print(
            f"  Second-moment release: enabled "
            f"(overhead={second_moment_arg if isinstance(second_moment_arg, float) else 'sqrt(3/2)'})"
        )
    print(
        f"  Target: ε={args.target_epsilon:.3f}, δ={args.target_delta:.2e} (n={global_train_size})"
    )
    print(f"  Noise multiplier: {noise_multiplier:.4f}")
    print(f"  Final ε (theoretical): {final_epsilon:.4f}")
    if args.audit:
        audit_result = run_audit(trainable_params)
        audit_eps = audit_result.epsilon_at(delta=args.target_delta)
        audit_auc = audit_result.auc()
        print(
            f"  Final ε (empirical):  {audit_eps:.4f}  ({audit_result.n_in} in, {audit_result.n_out} out)"
        )
        print(f"  Audit AUC:            {audit_auc:.4f}")
        print(f"  β @ α=0.01:           {audit_result.beta_at(alpha=0.01):.4f}")
        print(f"  β @ α=0.10:           {audit_result.beta_at(alpha=0.1):.4f}")
        if use_wandb:
            wandb.log(
                {
                    "privacy/epsilon_empirical": audit_eps,
                    "privacy/audit_auc": audit_auc,
                },
                step=global_step,
            )

    # Mark training complete and print profiler summary
    profiler, _ = profiler.mark("training_complete")
    summary_profiler = sync(profiler) if is_ddp else profiler
    print("\n" + summary_profiler.final_summary())
    print("\n" + summary_profiler.checkpoint_summary())

    # Record denoised eval metrics in the wandb summary so run-to-run
    # comparison uses min / EMA instead of the noisy final-step value.
    if use_wandb:
        wandb.run.summary["eval/loss_min"] = best_eval_loss
        if _is_cls and best_glue_score is not None:
            # The headline number for this run. Both the final and the max are
            # recorded: the max is what the LoRA-XS paper reports (best epoch,
            # then median over seeds), the final is what an honest fixed-budget
            # protocol reports, and the full-LoRA overfitting curve showed how far
            # apart those two can be.
            wandb.run.summary[f"eval/{glue_task.metric}"] = _last_eval_metric
            wandb.run.summary["eval/glue_score"] = _last_eval_metric
            wandb.run.summary["eval/glue_score_max"] = best_glue_score
            wandb.run.summary["eval/glue_score_max_step"] = best_glue_step
            wandb.run.summary["eval/glue_metric_name"] = glue_task.metric
        wandb.run.summary["eval/loss_min_step"] = best_eval_step
        if _ema_beta > 0:
            wandb.run.summary["eval/loss_ema"] = eval_loss_ema

    # Restore the best-eval checkpoint before saving / downstream eval.
    # trainable_params share storage with peft_model (functional detach +
    # in-place optimizer updates), so copy_ propagates to the live model.
    if best_snapshot is not None and is_main_process:
        if best_eval_step != global_step:
            print(
                f"\nRestoring best checkpoint (eval/loss={best_eval_loss:.4f} "
                f"@ step {best_eval_step}, vs final step {global_step})..."
            )
            for k, v in trainable_params.items():
                v.detach().copy_(best_snapshot[k].to(v.device, v.dtype))
        else:
            print(f"\nBest checkpoint IS the final step ({global_step}); no restore needed.")

    # Bits-per-byte on the eval set (high-SNR primary metric; per-example values
    # are logged so arms can be compared with paired bootstrap / sign-flip tests).
    if getattr(args, "eval_bpb", False) and is_main_process:
        import json as _json

        print(f"\nComputing BPB on {args.eval_bpb_samples} eval examples...")
        _t0 = time.time()
        _bpb, _bpb_list = eval_bpb(
            trainable_params,
            n_samples=args.eval_bpb_samples,
            microbatch=args.eval_bpb_microbatch,
        )
        _mean = sum(_bpb_list) / max(1, len(_bpb_list))
        _var = sum((x - _mean) ** 2 for x in _bpb_list) / max(1, len(_bpb_list) - 1)
        _sem = (_var ** 0.5) / max(1, len(_bpb_list)) ** 0.5
        print(
            f"  eval/bpb = {_bpb:.6f} bits/byte  (per-example mean {_mean:.6f} "
            f"± {_sem:.6f} SEM, n={len(_bpb_list)}, {time.time()-_t0:.0f}s)"
        )
        if use_wandb:
            wandb.run.summary["eval/bpb"] = _bpb
            wandb.run.summary["eval/bpb_per_example_mean"] = _mean
            wandb.run.summary["eval/bpb_per_example_sem"] = _sem
            wandb.run.summary["eval/bpb_n"] = len(_bpb_list)
            wandb.run.summary["eval/bpb_per_example_json"] = _json.dumps(
                [round(x, 6) for x in _bpb_list]
            )

    # Optional: dump per-layer core spectra for Rényi rank allocation (probe).
    # Pair with --max-steps N for a short warm-up; feed the JSON to
    # examples/compute_rank_allocation.py to build a --lora-xs-rank-pattern-json.
    if getattr(args, "dump_core_spectra", None) and is_main_process:
        import json as _json

        _pm = model._module if hasattr(model, "_module") else model
        _spectra: dict[str, list[float]] = {}
        for _name, _mod in _pm.named_modules():
            _R = getattr(_mod, "lora_xs_R", None)
            if _R is None or "default" not in _R:
                continue
            _entry = _R["default"]
            _w = getattr(_entry, "weight", _entry).detach().float()
            if _w.ndim != 2:
                continue
            _key = _name
            for _pfx in ("base_model.model.", "base_model."):
                if _key.startswith(_pfx):
                    _key = _key[len(_pfx):]
                    break
            _spectra[_key] = torch.linalg.svdvals(_w).cpu().tolist()
        with open(args.dump_core_spectra, "w") as _f:
            _json.dump(_spectra, _f)
        print(f"Dumped {len(_spectra)} per-layer core spectra to {args.dump_core_spectra}")
        # Also stash on the W&B run summary so a probe run's spectra can be read
        # back remotely (pods are ephemeral) to compute the allocation.
        if use_wandb:
            try:
                wandb.run.summary["probe/core_spectra_json"] = _json.dumps(_spectra)
                print("Logged probe/core_spectra_json to W&B summary")
            except Exception as _e:
                print(f"W&B spectra log failed: {_e}")

    # Save model and run downstream evaluation
    if args.output_dir and is_main_process:
        from pathlib import Path
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"\nSaving adapter to {args.output_dir}...")
        # Unwrap DP model to get the PEFT model
        peft_model = model._module if hasattr(model, "_module") else model
        peft_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Saved adapter + tokenizer to {args.output_dir}")

        if args.eval_humaneval or args.eval_mbpp:
            # Use peft_model directly without merging.
            # merge_and_unload paths failed differently for both LoRA-XS
            # (PEFT Parameter type mismatch) and LoRA ('Linear' object has
            # no attribute 'base_layer' — generate_answers expects PEFT
            # layer attribute). The unmerged path works for both.
            merged = peft_model.to(device).to(torch.bfloat16)

        if args.eval_humaneval and merged is not None:
            print("\n" + "=" * 60)
            print("Running HumanEval evaluation...")
            print("=" * 60)
            try:
                from lora_privacy.evaluation.code_eval import evaluate_humaneval

                results = evaluate_humaneval(
                    model=merged,
                    tokenizer=tokenizer,
                    batch_size=args.eval_batch_size or 4,
                    max_new_tokens=256,
                )
                print(f"\nHumanEval Results:")
                for k, v in results.items():
                    print(f"  {k}: {v:.4f}")
                    if use_wandb:
                        wandb.log({f"downstream/{k}": v}, step=global_step)
            except Exception as e:
                print(f"HumanEval evaluation failed: {e}")

        if args.eval_mbpp and merged is not None:
            print("\n" + "=" * 60)
            print("Running MBPP+ evaluation...")
            print("=" * 60)
            try:
                from lora_privacy.evaluation.code_eval import evaluate_mbpp

                results = evaluate_mbpp(
                    model=merged,
                    tokenizer=tokenizer,
                    batch_size=args.eval_batch_size or 4,
                    max_new_tokens=256,
                )
                print(f"\nMBPP+ Results:")
                for k, v in results.items():
                    print(f"  {k}: {v:.4f}")
                    if use_wandb:
                        wandb.log({f"downstream/{k}": v}, step=global_step)
            except Exception as e:
                print(f"MBPP evaluation failed: {e}")

    if use_wandb:
        wandb.finish()

    if is_ddp:
        dist.barrier(device_ids=[local_rank])
        _cleanup_distributed()

    return 0


if __name__ == "__main__":
    exit(main())
