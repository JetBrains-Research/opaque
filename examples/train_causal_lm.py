"""End-to-end DP-SGD LoRA training example for causal language models.

This example is designed as a production-style script (not a tutorial):
- clipping + noise + accounting always enabled
- adaptive clipping by default (--clipping-mode fixed|adaptive|auto)
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
import os
import sys
import time

import torch
import torch.distributed as dist
import torchopt
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model

from opaque.patches import apply_model_patches, apply_runtime_patches

apply_runtime_patches()

from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

import opaque.accounting as acc
import opaque.auditing as auditing
import opaque.dpsgd.accounting as dpsgd_acc
from opaque.accounting import calibration as cal, Accountant
from opaque.dpsgd.clipping import auto_clipped_grad, clipped_grad
from opaque.dpsgd.clipping import adaptive_clipped_grad
from opaque.distributed import sync
from opaque.distributed.gradients import sum_gradients_
from opaque.dpsgd.noise import gaussian_noise
from opaque.profiling import (
    perf_tracker,
    print_memory,
    reset_peak_memory,
)
from opaque.random import key, fold_in
from opaque.dpsgd.sampling import PoissonSampler
from opaque.distributed import local_shard
from opaque.functional import make_functional
from opaque.scheduling import (
    cosine_schedule,
    inverse_sqrt_schedule,
    linear_schedule,
    with_warmup,
)
from opaque.scheduling.types import Schedule
from opaque.types import (
    ClippedPytree,
    PerGroup,
    SecondMomentClippingOutput,
    SecondMomentNoiseOutput,
)
from opaque.dpsgd.clipping import per_group
import wandb


def _compile_grad_fn(grad_fn, *, backend, mode):
    """Caller-applied ``torch.compile`` of the DP grad transform.

    Compiles the whole ``vmap(grad(loss))`` + clipping step (functorch *inside*
    the compiled region — the supported, fusing pattern; ``vmap(grad)`` *outside*
    a compiled loss is the unsupported ``grad(compiled_fn)`` case). Tries
    ``fullgraph=True`` first and lazily falls back to ``fullgraph=False`` on the
    first graph break (the failure is lazy — it surfaces on first execution).
    """
    full = torch.compile(grad_fn, backend=backend, mode=mode, fullgraph=True)
    fallback = []

    def wrapper(*args, **kwargs):
        if fallback:
            return fallback[0](*args, **kwargs)
        try:
            return full(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 — any compile failure → fallback
            print(
                f"WARNING: torch.compile(fullgraph=True) failed "
                f"({type(e).__name__}: {e}); falling back to fullgraph=False."
            )
            fallback.append(
                torch.compile(grad_fn, backend=backend, mode=mode, fullgraph=False)
            )
            return fallback[0](*args, **kwargs)

    return wrapper


def _effective(value):
    """Extract scalar from float or PerGroup for logging/printing."""
    return value.effective if isinstance(value, PerGroup) else value


def _noise_stddev(max_norm, noise_multiplier, *, per_group=True):
    """Noise stddev: MSE-optimal per-group when available, isotropic otherwise."""
    if per_group and isinstance(max_norm, PerGroup):
        return ClippedPytree(pytree={}, max_norm=max_norm).noise_stddev_for(
            noise_multiplier=noise_multiplier
        )
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


def _log_private_second_moment() -> None:
    """Log paired-stream second-moment release.

    Privacy accounting is unchanged from the first-moment-only release —
    sensitivity-proportional Mahalanobis allocation makes the joint
    paired PLD identical to ``gaussian(nm)`` (or ``mf_gaussian(nm)``) at
    the same multiplier.
    """
    print("  Second moments: on (sensitivity-proportional allocation)")


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
        choices=["custom", "smoke", "mellum-kstack", "mellum2-kstack", "qwen-7b-kstack"],
        default="smoke",
        help="Apply preset configuration (custom=keep explicit args, smoke=quick test ~2min, mellum-kstack=Mellum-4b + KStack at ε=10 with adafactor @ 5e-5, qwen-7b-kstack=Qwen2.5-Coder-7B + KStack at ε=3 with adafactor @ 5e-4).",
    )

    model_group = parser.add_argument_group("model", "Model and tokenizer settings")
    model_group.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
        help="HuggingFace model name or local path",
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

    lora_group = parser.add_argument_group("lora", "LoRA adapter settings")
    lora_group.add_argument("--lora-r", type=int, default=4, help="LoRA rank")
    lora_group.add_argument("--lora-alpha", type=int, default=8, help="LoRA alpha")
    lora_group.add_argument(
        "--lora-modules",
        type=str,
        nargs="+",
        default=["c_attn", "c_proj"],
        help="Target module names for LoRA",
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
        default="adaptive",
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
        "--kernel-patches",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("OPAQUE_NO_KERNEL_PATCH", "0") != "1",
        help="Apply the opaque Triton speed kernels (rope/rms_norm/activation/"
        "fused-CE) to the model. On by default (auto-falls back to eager on "
        "non-CUDA hosts). --no-kernel-patches forces the eager baseline; the "
        "compat vmap-safety wrappers — including the load-bearing MoE experts "
        "patch — plus kv_cache and PEFT kernels stay on. Default also follows "
        "OPAQUE_NO_KERNEL_PATCH=1.",
    )
    dp_group.add_argument(
        "--grouped-moe",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force the grouped-GEMM MoE on/off (kernel-fused Triton on CUDA / "
        "torch._grouped_mm on MPS-CPU) independent of --kernel-patches. Default "
        "(unset) follows the kernels gate. Only affects MoE models (e.g. Mellum2); "
        "--no-grouped-moe runs the dense expert path.",
    )
    dp_group.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="torch.compile the DP per-example grad step (vmap(grad) + clipping). "
        "Off by default; a large speedup on GPU/MPS once warm (the first step "
        "pays the compile cost).",
    )
    dp_group.add_argument(
        "--torch-compile-backend",
        type=str,
        default="inductor",
        help="torch.compile backend (default: inductor).",
    )
    dp_group.add_argument(
        "--torch-compile-mode",
        type=str,
        default="default",
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        help="torch.compile mode (default: 'default').",
    )
    dp_group.add_argument(
        "--truncated-batch-size",
        type=int,
        default=None,
        help="Optional cap on per-step batch size (truncated Poisson). "
        "When set, the sampler caps each Poisson draw at this size and the "
        "accountant switches to the matching truncated Poisson-Gaussian PLD. "
        "Standard plain Poisson when omitted.",
    )
    dp_group.add_argument(
        "--noise-mechanism",
        type=str,
        choices=["gaussian", "bounded_gaussian"],
        default="gaussian",
        help="Noise mechanism: gaussian (standard, unbounded support) "
        "or bounded_gaussian (Chen and Hale, 2024; renormalized density on a "
        "per-coordinate interval).",
    )
    dp_group.add_argument(
        "--noise-bound",
        type=float,
        default=1.0,
        help="Symmetric absolute bound B for bounded_gaussian (per-coordinate "
        "support [-B, B]; same units as the gradient / clip norm). Ignored "
        "for standard gaussian.",
    )
    dp_group.add_argument(
        "--second-moment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Private squared-gradient stream alongside gradients. "
        "Off by default; pass ``--second-moment`` to enable.  Requires an "
        "optimizer that consumes ``SecondMomentNoiseOutput`` "
        "(adam/adamw/ademamix/rmsprop/radam/adadelta); incompatible "
        "combinations are warned-and-disabled (not rejected).  Joint noise "
        "allocation is sensitivity-proportional; privacy accounting is "
        "gaussian(nm) — same as first-moment-only.",
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
        "--audit-method",
        choices=["gdp", "eps_delta"],
        default="gdp",
        help="Which audit method's ε to report ('gdp' = μ-GDP, recommended for Gaussian-DP mechanisms like DP-SGD; 'eps_delta' = mechanism-agnostic (ε, δ)-DP fallback)",
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
    elif args.preset == "mellum2-kstack":
        # Mellum2-12B-A2.5B (MoE) + KStack at ε=10. LoRA on attention projections
        # only; the routed experts are stacked nn.Parameter weights PEFT can't adapt.
        _set("model_name", "JetBrains/Mellum2-12B-A2.5B-Base")
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
        _set("lora_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
        _set("dtype", "bfloat16")
        _set("microbatch_size", 8)
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
    elif args.preset == "custom":
        # Keep all user-provided/default CLI arguments unchanged.
        pass

    # --microbatch-size 0 means "no microbatching" (full-batch vmap).
    # Needed because argparse type=int can't accept None on CLI to override presets.
    if args.microbatch_size == 0:
        args.microbatch_size = None

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

    # μ-GDP auditing has no meaningful answer at δ = 0 (pure DP is incompatible
    # with Gaussian DP).  Fail fast instead of crashing inside the audit.
    if args.audit and args.audit_method == "gdp" and args.target_delta <= 0:
        raise SystemExit(
            "--audit-method gdp requires --target-delta > 0 "
            f"(got {args.target_delta}); use --audit-method eps_delta for pure DP."
        )

    if is_main_process:
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

    # Load model config and disable dropout
    print(f"\nLoading model: {args.model_name}...")
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

    tracker = perf_tracker(device)

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    except TypeError as exc:
        if "unexpected keyword argument 'dtype'" not in str(exc):
            raise
        model_kwargs.pop("dtype")
        model_kwargs["torch_dtype"] = torch_dtype
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model = model.to(device)
    print_memory(device, "After model load")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA
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
    # Loss is the only consumer of the forward output in this script, so the
    # default opts into the fused linear+CE kernel (skips ``lm_head`` and returns
    # ``logits=None`` on the fast path — see ``apply_model_patches`` docs).
    #
    # ``--no-kernel-patches`` passes ``kernels=False`` for an eager baseline: it
    # disables ONLY the Triton speed kernels (rope / rms_norm / activation /
    # cross-entropy) and the fused linear-CE. The ``compat`` vmap-safety wrappers
    # stay on — crucially the stacked-MoE ``experts`` patch, which lives in the
    # compat bucket (not ``kernels``) because HF's experts forward isn't
    # ``vmap(grad)``-able and DP-SGD breaks without it — as do ``kv_cache`` and
    # the PEFT kernels. The global runtime patches above are untouched either way.
    # ``--grouped-moe`` overrides the grouped-GEMM MoE gate independent of
    # --kernel-patches (default None = follow the kernels gate). Lets the eager
    # baseline keep dense MoE while a kernels-off run still exercises grouped, and
    # vice-versa — used to measure grouped-vs-dense MoE on MPS/CPU.
    moe_kw = {} if args.grouped_moe is None else {"grouped_moe": args.grouped_moe}
    if args.kernel_patches:
        apply_model_patches(
            model, kernels=True, fused_linear_cross_entropy=True, **moe_kw
        )
    else:
        print("Kernel patches: DISABLED (eager baseline; compat/MoE/PEFT stay on)")
        apply_model_patches(
            model, kernels=False, fused_linear_cross_entropy=False, **moe_kw
        )
    model.print_trainable_parameters()
    print_memory(device, "After LoRA")

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
    truncated_batch_size = args.truncated_batch_size
    sample_rate = args.batch_size / global_train_size
    if use_parallel_poisson:
        sample_rate /= world_size

    expected_steps_per_epoch = int(global_train_size / args.batch_size)

    print("\nPoisson sampling setup:")
    if use_parallel_poisson:
        print(f"  Mode: parallel_poisson (no shard, world_size={world_size})")
    print(f"  Sampler: poisson")
    if truncated_batch_size is not None:
        print(f"  Truncated batch size (cap): {truncated_batch_size}")
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
    print_memory(device, "After functional conversion")

    def merged_params(trainable):
        return {**frozen_params, **trainable}

    # Hoist ``pad_token_id`` to a plain int. Reading ``tokenizer.pad_token_id``
    # *inside* the per-example loss hits the tokenizer's C-level ``__getattr__``
    # on every call — which Dynamo can't trace, so torch.compile graph-breaks
    # there (``fullgraph=True`` bails). It's a constant; capture it once outside
    # the hot, vmap(grad)-traced path.
    pad_token_id = tokenizer.pad_token_id

    # Define per-example loss
    def per_example_loss_fn(trainable, input_ids):
        # Mask pad positions to ``-100`` so training CE scores only real tokens —
        # same masking the eval path and DPTrainer's collator apply. Without it the
        # manual loop trains on unmasked labels while DPTrainer trains on masked
        # ones, drifting ``train/loss`` under identical DP math. ``vmap``-safe.
        labels = torch.where(input_ids == pad_token_id, -100, input_ids)
        output = fmodel(merged_params(trainable), input_ids, labels=labels)
        return output.loss

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

    def _audit_method(estimate):
        """Pick the audit-method object on `estimate` per ``args.audit_method``."""
        return estimate.gdp() if args.audit_method == "gdp" else estimate.eps_delta()

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

    def eval_loss(trainable):
        """Token-weighted CE over the eval set (pad tokens masked to ``-100``).

        Returns ``float('nan')`` when the eval set has zero scoring tokens
        (empty ``--num-eval-samples`` or all examples shorter than 2 tokens).
        """
        with torch.no_grad():
            total_loss = 0.0
            total_tokens = 0
            for (input_ids,) in eval_loader:
                labels = input_ids.clone()
                labels[labels == pad_token_id] = -100
                output = fmodel(merged_params(trainable), input_ids, labels=labels)
                num_tokens = int((labels[..., 1:] != -100).sum().item())
                if num_tokens == 0:
                    continue
                total_loss += output.loss.item() * num_tokens
                total_tokens += num_tokens
            if total_tokens == 0:
                return float("nan")
            return total_loss / total_tokens

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
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Learning rate: {args.learning_rate}")
    if isinstance(clip_norm, PerGroup):
        print(f"  Clip norm: per-group (effective={clip_norm.effective:.3f})")
    else:
        print(f"  Clip norm: {clip_norm}")
    print(f"  Noise mechanism: {args.noise_mechanism}")
    if args.noise_mechanism != "gaussian":
        print(f"  Noise bound: ±{args.noise_bound}")
    print(f"  Microbatch size: {args.microbatch_size}")
    print(f"  Clipping mode: {args.clipping_mode}")
    if args.clipping_mode == "auto":
        print(f"  AUTO-S gamma: {args.auto_clipping_gamma}")
    print(f"  Eval steps: {args.eval_steps}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Expected total steps: ~{args.num_epochs * expected_steps_per_epoch}")

    # Cross-flag validation for --second-moment: warn-and-disable on
    # mismatch instead of raising.  The squared-gradient stream is
    # auxiliary; silently dropping to single-stream noise on an
    # incompatible optimizer beats failing the run outright.
    _SECOND_MOMENT_OPTIMIZERS = frozenset(
        {"adam", "adamw", "ademamix", "rmsprop", "radam", "adadelta"}
    )
    use_second_moment = bool(args.second_moment)
    if use_second_moment and args.optimizer not in _SECOND_MOMENT_OPTIMIZERS:
        print(
            f"\nWARNING: --second-moment requires an optimizer that consumes "
            f"SecondMomentNoiseOutput ({sorted(_SECOND_MOMENT_OPTIMIZERS)}); "
            f"got --optimizer {args.optimizer!r}.  Disabling --second-moment "
            f"for this run."
        )
        use_second_moment = False

    # Create gradient function based on clipping mode.
    if args.clipping_mode == "adaptive":
        # ``second_moment`` flows through ``**clipped_grad_kwargs`` to the
        # inner ``clipped_grad`` call; the adaptive threshold update reads
        # the first-stream gradient norms regardless of paired-stream output.
        grad_fn, clip_state = adaptive_clipped_grad(
            per_example_loss_fn,
            argnums=0,
            batch_argnums=(1,),
            initial_clipping_norm=clip_norm,
            target_quantile=1.0 - args.target_clipping_rate,
            clipping_norm_max=args.clipping_norm_max,
            microbatch_size=args.microbatch_size,
            return_aux=True,
            key=key(args.seed),
            normalize_by=args.batch_size,
            second_moment=use_second_moment,
        )
    elif args.clipping_mode == "auto":
        grad_fn, clip_state = auto_clipped_grad(
            per_example_loss_fn,
            argnums=0,
            batch_argnums=(1,),
            R=clip_norm,
            gamma=args.auto_clipping_gamma,
            normalize_by=args.batch_size,
            microbatch_size=args.microbatch_size,
            return_aux=True,
            second_moment=use_second_moment,
        )
    else:
        grad_fn, clip_state = clipped_grad(
            per_example_loss_fn,
            argnums=0,
            batch_argnums=(1,),
            clipping_norm=clip_norm,
            normalize_by=args.batch_size,
            microbatch_size=args.microbatch_size,
            return_aux=True,
            second_moment=use_second_moment,
        )

    # Caller-applied torch.compile of the whole DP grad transform (opt-in).
    if args.torch_compile:
        grad_fn = _compile_grad_fn(
            grad_fn,
            backend=args.torch_compile_backend,
            mode=args.torch_compile_mode,
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
    # Bounded Gaussian noise (Chen and Hale, 2024) confines per-coordinate
    # support but accounting collapses to ordinary Gaussian at training
    # scale (ℓ₂-ball clip, not a product of intervals), so we use
    # dpsgd_acc.gaussian() for accounting either way.
    _num_groups = len(clip_norm.values) if isinstance(clip_norm, PerGroup) else 1
    if args.noise_multiplier == 0:
        mechanism = lambda nm: acc.nonprivate()
    elif args.noise_mechanism == "bounded_gaussian":
        mechanism = dpsgd_acc.gaussian
    else:
        mechanism = dpsgd_acc.gaussian

    if args.clipping_mode == "adaptive":
        _base_mechanism = mechanism
        mechanism = lambda nm, ebs=args.batch_size, ng=_num_groups: dpsgd_acc.adaclip(
            _base_mechanism(nm), expected_batch_size=ebs, num_groups=ng
        )
    # No paired-stream wrap: joint Mahalanobis allocation makes
    # the second moment release "free" at the runtime σ allocation
    # level; calibration uses the same gaussian(nm) PLD as the
    # first-moment-only release.

    _unamplified = mechanism
    if truncated_batch_size is not None:
        mechanism = lambda nm: dpsgd_acc.poisson(
            _unamplified(nm),
            sample_rate=sample_rate,
            truncated_batch_size=truncated_batch_size,
            dataset_size=global_train_size,
        )
    elif use_parallel_poisson:
        mechanism = lambda nm: dpsgd_acc.parallel_poisson(
            _unamplified(nm),
            sample_rate=sample_rate,
            num_workers=world_size,
        )
    else:
        mechanism = lambda nm: dpsgd_acc.poisson(
            _unamplified(nm), sample_rate=sample_rate
        )

    # Calibrate noise multiplier from target privacy budget.
    if args.noise_multiplier is not None:
        noise_multiplier = args.noise_multiplier
        print(
            f"\nUsing fixed noise multiplier: {noise_multiplier:.4f} (skipping calibration)"
        )
        if use_second_moment:
            _log_private_second_moment()
    else:
        print("\nCalibrating privacy parameters...")
        if use_parallel_poisson:
            print(f"  Accounting: parallel_poisson (world_size={world_size})")
        print(f"  Noise mechanism: {args.noise_mechanism}")
        if args.noise_mechanism == "bounded_gaussian":
            print(f"  Noise bound: ±{args.noise_bound}")
        if use_second_moment:
            _log_private_second_moment()
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
        raise ValueError(f"--lr-min-ratio must be in [0, 1], got {args.lr_min_ratio}")
    peak_lr = args.learning_rate
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
        from opaque.optimizers import sgd

        base_opt = sgd(lr=lr_for_opt, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        from opaque.optimizers import adamw

        base_opt = adamw(
            lr=lr_for_opt,
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

    opt_state = base_opt.init(trainable_params)
    accounting = Accountant()

    # Noise functions consume ClippedPytree metadata directly and return
    # NoisedPytree updates carrying the realized per-step stddev.
    initial_bound = clip_norm / args.batch_size
    if args.noise_mechanism == "bounded_gaussian":
        # Pass ``bound`` unconditionally — at ``noise_multiplier=0`` the
        # bounded path clamps the input to the interval (vs. the unbounded
        # path which returns it unchanged), so the mechanism stays
        # consistent for the user's chosen flag.
        noise_fn, noise_state = gaussian_noise(
            noise_multiplier=noise_multiplier,
            bound=args.noise_bound,
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

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 80)
        print("Creating poisson sampler...")

        # Create sampler for this epoch
        epoch_sampler = PoissonSampler(
            train_dataset,
            sample_rate=sample_rate,
            truncated_batch_size=truncated_batch_size,
            n_steps=expected_steps_per_epoch,
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
            (input_ids,) = batch

            # === Accounting (data-independent, before execution) ===
            accounting |= mechanism(noise_multiplier)

            batch_size = len(input_ids)

            # === Execution ===
            with tracker.train(batch_size=batch_size) as sp:
                # Compute clipped gradients (handles empty batches via library)
                with offload_ctx:
                    (grads_tuple, aux), clip_state = grad_fn(
                        trainable_params, input_ids, state=clip_state
                    )
                if is_ddp:
                    clip_state, aux = sync(clip_state, aux)
                    sum_gradients_(grads_tuple)
                sp.mark("clip")

                step_clip_norm = _step_clip_norm(grads_tuple)
                noisy_grads, noise_state = noise_fn(grads_tuple, noise_state)
                noise_stddev = _step_noise_stddev(noisy_grads)
                if is_ddp:
                    noise_state = sync(noise_state)
                sp.mark("noise")

                updates, opt_state = base_opt.update(
                    noisy_grads, opt_state, params=trainable_params
                )
                trainable_params = torchopt.apply_updates(trainable_params, updates)
                sp.mark("optimizer")

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

            global_step += 1

            # === Logging (every log_steps) ===
            if global_step % args.log_steps == 0:
                if use_wandb:
                    current_lr = (
                        float(lr_for_opt(global_step))
                        if callable(lr_for_opt)
                        else float(lr_for_opt)
                    )
                    wb_metrics = {
                        "train/loss": avg_loss,
                        "train/batch_size": batch_size,
                        "train/clipping_norm": _effective(step_clip_norm),
                        "train/clip_rate": clip_rate,
                        "train/grad_norm_mean": mean_grad_norm,
                        "train/clipped_grad_norm_mean": aux.clipped_grad_norms.mean().item(),
                        "train/noise_std": _effective(noise_stddev),
                        "train/lr": current_lr,
                        **tracker.train.last.to_dict(prefix="train/"),
                    }
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
                            if isinstance(noise_stddev, PerGroup):
                                wb_metrics[f"group/noise_std/{gname}"] = (
                                    noise_stddev.values[gname]
                                )
                    wandb.log(wb_metrics, step=global_step)

                last = tracker.train.last
                print(
                    f"Step {global_step:4d} [E{epoch + 1} S{step_idx + 1:3d}/{expected_steps_per_epoch:3d}] | "
                    f"BS: {batch_size} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Clip: norm={_effective(step_clip_norm):.3f}, rate={clip_rate:.1%} | "
                    f"GradNorm: μ={mean_grad_norm:.3f} | "
                    f"Noise: σ={_effective(noise_stddev):.4f} | "
                    f"Time: {last.step_time_sec:.2f}s | Mem: {last.memory_peak_gb:.1f}GB"
                )

            # Expensive operations (eval + privacy + audit) every eval_steps
            if global_step % args.eval_steps == 0:
                current_eval_loss = eval_loss(trainable_params)
                # Cache PLD before eval so it serves as opaque boundary
                accounting = acc.cached(accounting)
                epsilon = accounting.epsilon_at(args.target_delta)

                metrics = {
                    "eval/loss": current_eval_loss,
                    "privacy/epsilon": epsilon,
                }
                eval_msg = f"  → Eval: loss={current_eval_loss:.4f}, ε={epsilon:.3f}"

                if args.audit:
                    audit_estimate = run_audit(trainable_params)
                    audit_eps = _audit_method(audit_estimate).epsilon_at(
                        delta=args.target_delta
                    )
                    audit_auc = audit_estimate.attack_auc()
                    metrics["privacy/epsilon_audit"] = audit_eps
                    metrics["privacy/audit_auc"] = audit_auc
                    eval_msg += (
                        f", ε_audit[{args.audit_method}]={audit_eps:.4f}"
                        f", AUC={audit_auc:.4f}"
                    )

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
    if truncated_batch_size is not None:
        print(
            f"  Accounting: truncated_poisson (cap={truncated_batch_size}, n={global_train_size})"
        )
    elif use_parallel_poisson:
        print(f"  Accounting: parallel_poisson (world_size={world_size})")
    if use_second_moment:
        _log_private_second_moment()
    print(
        f"  Target: ε={args.target_epsilon:.3f}, δ={args.target_delta:.2e} (n={global_train_size})"
    )
    print(f"  Noise multiplier: {noise_multiplier:.4f}")
    print(f"  Final ε (theoretical): {final_epsilon:.4f}")
    if args.audit:
        audit_result = run_audit(trainable_params)
        audit_eps = _audit_method(audit_result).epsilon_at(delta=args.target_delta)
        audit_auc = audit_result.attack_auc()
        print(
            f"  Final ε (audit, {args.audit_method}): {audit_eps:.4f}"
            f"  ({audit_result.n_in} in, {audit_result.n_out} out)"
        )
        print(f"  Audit AUC:            {audit_auc:.4f}")
        # Empirical attack ROC β (1 − TPR at given FPR); independent of the
        # audit method, hence read from OneRunEstimate rather than the method.
        print(f"  Attack β @ α=0.01:    {audit_result.attack_beta_at(alpha=0.01):.4f}")
        print(f"  Attack β @ α=0.10:    {audit_result.attack_beta_at(alpha=0.1):.4f}")
        if use_wandb:
            wandb.log(
                {
                    "privacy/epsilon_audit": audit_eps,
                    "privacy/audit_auc": audit_auc,
                },
                step=global_step,
            )

    synced = sync(tracker) if is_ddp else tracker
    print("\nPerformance:")
    print(f"  Throughput: {synced.train.samples_per_second:.1f} samples/s")
    print(f"  Steps/s: {synced.train.steps_per_second:.2f}")
    print(f"  Peak memory: {synced.train.max_peak_memory_gb:.2f} GB")

    if use_wandb:
        wandb.finish()

    if is_ddp:
        dist.barrier(device_ids=[local_rank])
        _cleanup_distributed()

    return 0


if __name__ == "__main__":
    exit(main())
