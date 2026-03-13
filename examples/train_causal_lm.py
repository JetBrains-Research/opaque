"""End-to-end DP-SGD LoRA training example for causal language models.

This example is designed as a production-style script (not a tutorial):
- clipping + noise + accounting always enabled
- adaptive clipping enabled by default
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
    --model_name "JetBrains/Mellum-4b-base" \\
    --dataset "JetBrains/KStack" \\
    --dataset_text_field "content" \\
    --num_train_samples 50000 \\
    --num_eval_samples 1000 \\
    --num_epochs 3 \\
    --batch_size 32 \\
    --eval_steps 50 \\
    --target_epsilon 10.0 \\
    --learning_rate 5e-5 \\
    --lora_r 16 --lora_alpha 32 \\
    --max_seq_len 1024 \\
    --lora_modules q_proj k_proj v_proj o_proj \\
    --audit --audit_canaries 1000 \\
    --no_wandb
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
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

import opaque.accounting as acc
import opaque.auditing as auditing
from opaque.accounting import calibration as cal, Accountant
from opaque.clipping import adaptive_clipped_grad, clipped_grad
from opaque.compat.transformers import is_kernel_patched
from opaque.distributed import sum_gradients_, sync
from opaque.noise import gaussian_noise
from opaque.profiling import StepTimer, TrainingProfiler, print_memory, reset_peak_memory
from opaque.random import key, fold_in
from opaque.sampling import PoissonSampler
from opaque.sampling.distributed import local_shard
from opaque.utils import make_functional
import wandb


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

    patched = is_kernel_patched()
    if patched:
        return "enabled", "global kernel patches applied"
    return "partial", "kernel patch state unavailable"


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
        print("  Note: MPS uses compatibility fallbacks when CUDA-only kernels are unavailable.")


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
        choices=["custom", "smoke", "mellum-kstack"],
        default="smoke",
        help="Apply preset configuration (custom=keep explicit args, smoke=quick test ~2min, mellum-kstack=full production).",
    )

    model_group = parser.add_argument_group("model", "Model and tokenizer settings")
    model_group.add_argument(
        "--model_name",
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
        "--sdpa_backend",
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
        "--dataset_subset",
        "--dataset_name",
        dest="dataset_subset",
        type=str,
        default=None,
        help="Optional dataset subset (HF load_dataset 'name' argument), e.g. 'stage1-auto-format'.",
    )
    data_group.add_argument(
        "--dataset_split", type=str, default="train", help="Dataset split for training"
    )
    data_group.add_argument(
        "--dataset_text_field",
        type=str,
        default="text",
        help="Field containing text",
    )
    data_group.add_argument(
        "--num_train_samples",
        type=int,
        default=5000,
        help="Number of training examples (default: 5000 for smoke test)",
    )
    data_group.add_argument(
        "--num_eval_samples",
        "--num_eval_samples_alt",
        dest="num_eval_samples",
        type=int,
        default=100,
        help="Number of samples for periodic eval-loss reporting (batched)",
    )
    data_group.add_argument(
        "--max_seq_len", type=int, default=512, help="Maximum sequence length"
    )

    train_group = parser.add_argument_group("training", "Training loop settings")
    train_group.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Expected batch size for Poisson sampling (determines sample_rate)"
    )
    train_group.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation (default: same as batch_size, can be larger since no privacy needed)"
    )
    train_group.add_argument(
        "--num_epochs", type=int, default=3, help="Number of epochs"
    )
    train_group.add_argument(
        "--learning_rate", type=float, default=1.0e-5, help="Learning rate"
    )
    train_group.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["sgd", "adam"],
        help="Optimizer",
    )
    train_group.add_argument(
        "--log_steps",
        type=int,
        default=1,
        help="Log training metrics every N steps",
    )
    train_group.add_argument(
        "--eval_steps",
        type=int,
        default=10,
        help="Log eval loss and privacy every N steps",
    )
    train_group.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum training steps (overrides num_epochs if set)",
    )
    train_group.add_argument("--seed", type=int, default=42, help="Random seed")
    train_group.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable gradient checkpointing for memory savings (trades compute for memory)",
    )
    train_group.add_argument(
        "--cpu_offload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Offload saved tensors to CPU via save_on_cpu (works with or without checkpointing)",
    )

    lora_group = parser.add_argument_group("lora", "LoRA adapter settings")
    lora_group.add_argument("--lora_r", type=int, default=4, help="LoRA rank")
    lora_group.add_argument("--lora_alpha", type=int, default=8, help="LoRA alpha")
    lora_group.add_argument(
        "--lora_modules",
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
        "Use --no_shard to replicate full dataset on each rank "
        "(uses parallel_poisson accounting).",
    )
    dp_group.add_argument(
        "--clip_norm",
        type=float,
        default=1.0,
        help="Clip norm (fixed mode) or starting clip norm (adaptive mode)",
    )
    dp_group.add_argument(
        "--adaptive_clipping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use adaptive clipping (default: True)",
    )
    dp_group.add_argument(
        "--target_clip_rate",
        type=float,
        default=0.5,
        help="Target clipping rate for adaptive clipping",
    )
    dp_group.add_argument(
        "--clip_norm_max",
        type=float,
        default=10.0,
        help="Maximum clip norm in adaptive mode",
    )
    dp_group.add_argument(
        "--microbatch_size",
        type=int,
        default=None,
        help="Microbatch size passed to clipped_grad/adaptive_clipped_grad (None=process full batch with vmap, faster but more memory)",
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
        "--target_epsilon",
        type=float,
        default=3.0,
        help="Target epsilon used to calibrate noise_multiplier",
    )
    privacy_group.add_argument(
        "--target_delta",
        type=float,
        default=None,
        help="Target delta for DP accounting. Default: 1/n² where n = training set size.",
    )
    privacy_group.add_argument(
        "--calibration_min",
        type=float,
        default=0.11,
        help="Lower bound for noise calibration search",
    )
    privacy_group.add_argument(
        "--calibration_max",
        type=float,
        default=1.19,
        help="Upper bound for noise calibration search",
    )
    privacy_group.add_argument(
        "--calibration_tolerance",
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
        "--audit_canaries",
        type=int,
        default=1000,
        help="Number of canaries for one-run auditing",
    )
    audit_group.add_argument(
        "--audit_batch_size",
        type=int,
        default=None,
        help="Batch size for auditing scoring (default: same as microbatch_size; forward-only so less memory than training)",
    )

    tracking_group = parser.add_argument_group("tracking", "Experiment tracking (W&B)")
    tracking_group.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable experiment tracking (wandb is enabled by default, offline if no credentials)",
    )
    tracking_group.add_argument(
        "--wandb_project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "opaque"),
        help="W&B project name (default: WANDB_PROJECT env var or 'opaque')",
    )
    tracking_group.add_argument(
        "--wandb_run_name",
        type=str,
        default=os.environ.get("WANDB_NAME") or os.environ.get("RUN_NAME"),
        help="Run name (default: WANDB_NAME, then RUN_NAME env var, or auto-generated from model and hyperparameters)",
    )
    tracking_group.add_argument(
        "--wandb_entity",
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
            if any(token == opt or token.startswith(f"{opt}=") for token in argv_tokens):
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
        # mb=32 gives same speed but 108 GB. With --gradient_checkpointing, mb=32+ fits easily.
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
        _set("lora_modules", [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ])
        _set("dtype", "bfloat16")
        _set("microbatch_size", 16)
    elif args.preset == "custom":
        # Keep all user-provided/default CLI arguments unchanged.
        pass

    return args


def main():
    args = parse_args()

    is_ddp, rank, world_size, local_rank = _init_distributed()
    is_main_process = rank == 0

    # Set eval_batch_size to batch_size if not specified
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size

    # Set audit_batch_size to microbatch_size if not specified (forward-only, so at least as cheap)
    if args.audit_batch_size is None:
        args.audit_batch_size = args.microbatch_size or args.batch_size

    if is_main_process:
        print("=" * 80)
        print("DP-SGD LoRA Training for Causal Language Models")
        print("=" * 80)

    # Initialize wandb (enabled by default, offline if no credentials)
    use_wandb = (not args.no_wandb) and is_main_process
    if use_wandb:
        # Generate default run name from key parameters if not specified
        if args.wandb_run_name is None:
            model_short = args.model_name.split('/')[-1]
            run_name = f"{model_short}_n{args.num_train_samples}_e{args.num_epochs}_b{args.batch_size}_eps{args.target_epsilon}_lr{args.learning_rate}"
        else:
            run_name = args.wandb_run_name

        # Offline by default; set WANDB_MODE=online (or WANDB_API_KEY) to sync
        if not os.environ.get("WANDB_MODE"):
            os.environ["WANDB_MODE"] = "online" if os.environ.get("WANDB_API_KEY") else "offline"
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
            print(f"Distributed mode: rank={rank}/{world_size}, local_rank={local_rank}")

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
    _print_runtime_mode_report(device, device_name, dtype_name, torch_dtype, dtype_warning)

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
    model.print_trainable_parameters()
    profiler, _ = profiler.mark("lora_applied")
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
    print(f"\nPreparing {args.num_eval_samples} eval + {args.num_train_samples} train samples...")
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
        desc="Tokenizing eval"
    )
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_cols_to_remove,
        desc="Tokenizing train"
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
    # Poisson: each example independently sampled with probability sample_rate each step
    # sample_rate is defined globally (across all ranks).
    sample_rate = args.batch_size / global_train_size

    # Expected number of steps to process full dataset with Poisson sampling
    # = 1 / sample_rate (since we sample sample_rate fraction each step)
    expected_steps_per_epoch = int(1.0 / sample_rate)

    print("\nPoisson sampling setup:")
    if use_parallel_poisson:
        print(f"  Mode: parallel_poisson (no shard, world_size={world_size})")
    print(f"  Sample rate: {sample_rate:.6f}")
    print(f"  Expected batch size: {args.batch_size}")
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
        print(f"CPU offload: enabled (save_on_cpu, works {'with' if args.gradient_checkpointing else 'without'} checkpointing)")

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

    # Define per-example loss
    def per_example_loss_fn(trainable, input_ids):
        output = fmodel(merged_params(trainable), input_ids, labels=input_ids)
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
        print(f"  Reference scores: mean={audit_ref_scores.mean():.4f}, std={audit_ref_scores.std():.4f}")

    def eval_loss(trainable):
        """Compute eval loss using DataLoader."""
        with torch.no_grad():
            total_loss = 0.0
            total_tokens = 0

            for (input_ids,) in eval_loader:
                loss = per_example_loss_fn(trainable, input_ids)
                total_loss += loss.item() * len(input_ids)
                total_tokens += len(input_ids)

            return total_loss / total_tokens

    # Setup optimizer
    print("\nSetting up DP-SGD training...")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Clip norm: {args.clip_norm}")
    print(f"  Microbatch size: {args.microbatch_size}")
    print(f"  Adaptive clipping: {args.adaptive_clipping}")
    print(f"  Eval steps: {args.eval_steps}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Expected total steps: ~{args.num_epochs * expected_steps_per_epoch}")

    if args.optimizer == "adam":
        base_opt = torchopt.adam(lr=args.learning_rate)
    elif args.optimizer == "sgd":
        base_opt = torchopt.sgd(lr=args.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # Create gradient function (adaptive or fixed clipping)
    if args.adaptive_clipping:
        grad_fn, clip_state = adaptive_clipped_grad(
            per_example_loss_fn,
            argnums=0,
            batch_argnums=(1,),
            initial_clip_norm=args.clip_norm,
            target_quantile=1.0 - args.target_clip_rate,
            clip_norm_max=args.clip_norm_max,
            microbatch_size=args.microbatch_size,
            return_aux=True,
            key=key(args.seed),
        )
    else:
        grad_fn, clip_state = clipped_grad(
            per_example_loss_fn,
            argnums=0,
            batch_argnums=(1,),
            l2_clip_norm=args.clip_norm,
            microbatch_size=args.microbatch_size,
            return_aux=True,
        )

    opt_state = base_opt.init(trainable_params)

    # Calibrate noise multiplier from target privacy budget
    # sample_rate already computed above
    total_steps = args.num_epochs * expected_steps_per_epoch

    # Compute delta from training set size: δ = 1/n^1.1 (keeps δ below 1/n while
    # being less conservative than the previous 1/n² heuristic on smaller runs).
    if args.target_delta is None:
        args.target_delta = 1.0 / (global_train_size ** 1.1)
    if use_wandb:
        wandb.config.update({"target_delta": args.target_delta}, allow_val_change=True)

    print("\nCalibrating privacy parameters...")
    if use_parallel_poisson:
        print(f"  Accounting: parallel_poisson (world_size={world_size})")
    print(f"  δ = {args.target_delta:.2e} (n={global_train_size})")
    print(f"  Total steps: {total_steps}")
    print(f"  Sample rate: {sample_rate:.6f}")
    print(f"  Target: ε={args.target_epsilon}, δ={args.target_delta:.2e}")
    print("  (This may take 1-3 minutes...)")

    start_time = time.time()
    budget = cal.epsilon_budget(args.target_epsilon, delta=args.target_delta)
    if use_parallel_poisson:
        process_fn = lambda nm: acc.parallel_poisson(
            acc.gaussian(nm), sample_rate=sample_rate, num_workers=world_size,
        ) * total_steps
    else:
        process_fn = lambda nm: acc.poisson(
            acc.gaussian(nm), sample_rate=sample_rate,
        ) * total_steps
    calibration = cal.calibrate(
        budget,
        process_fn,
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

    # Accounting (all pld() calls automatically cached with maxsize=8)
    # Using acc.cached() here increases cache to maxsize=16 and creates merge barrier
    accounting = Accountant()
    if use_parallel_poisson:
        step_process = acc.cached(acc.parallel_poisson(
            acc.gaussian(noise_multiplier), sample_rate=sample_rate, num_workers=world_size,
        ))
    else:
        step_process = acc.cached(acc.poisson(acc.gaussian(noise_multiplier), sample_rate))

    # Initialize noise function
    noise_fn, noise_state = gaussian_noise(stddev=noise_multiplier * clip_state.sensitivity(), key=key(args.seed))

    # Training loop
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    losses = []
    clip_norms_history = []
    clip_rates_history = []
    global_step = 0

    # Reset peak memory before training to get accurate training peak
    reset_peak_memory(device)
    profiler, _ = profiler.mark("training_start")
    print_memory(device, "Before training")

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 80)
        print("Creating Poisson sampler...")

        # Create Poisson sampler for this epoch
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
            (input_ids,) = batch

            # Accounting update (must happen even for empty batches)
            accounting |= step_process

            # Skip if no examples sampled (rare but possible with Poisson)
            if len(input_ids) == 0:
                continue

            # Time the training step using profiler
            step_timer = StepTimer(device, batch_size=len(input_ids))
            with step_timer:
                # Compute clipped gradients (with state passing)
                with offload_ctx:
                    (grads_tuple, aux), clip_state = grad_fn(
                        trainable_params, input_ids, state=clip_state
                    )
                current_clip_norm = clip_state.clip_norm

                if is_ddp:
                    clip_state, aux = sync(clip_state, aux)
                    current_clip_norm = clip_state.clip_norm
                    sum_gradients_(grads_tuple)

                # Add Gaussian noise
                stddev = noise_multiplier * clip_state.sensitivity()
                noisy_grads, noise_state = noise_fn(grads_tuple, noise_state)
                if is_ddp:
                    noise_state = sync(noise_state)

                # Optimizer step (no adapter wrapper - optimizer used directly)
                updates, opt_state = base_opt.update(
                    noisy_grads, opt_state, params=trainable_params
                )

                # Apply updates
                trainable_params = torchopt.apply_updates(trainable_params, updates)

            profiler = profiler.add_step(step_timer)

            # Extract metrics from aux
            avg_loss = aux.loss_values.mean().item()
            mean_grad_norm = aux.grad_norms.mean().item()
            clipped_grad_norm_mean = aux.clipped_grad_norms.mean().item()
            clip_rate = aux.clipping_rate

            losses.append(avg_loss)
            clip_norms_history.append(current_clip_norm)
            clip_rates_history.append(clip_rate)

            global_step += 1

            # Log training metrics every log_steps
            if global_step % args.log_steps == 0:
                num_clipped = int(clip_rate * len(aux.grad_norms))
                log_profiler = sync(profiler) if is_ddp else profiler
                profiler = log_profiler  # flush pending steps so next sync covers only new records
                perf_metrics = profiler.current_metrics()

                # W&B logging
                if use_wandb:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/clip_norm": current_clip_norm,
                        "train/clip_rate": clip_rate,
                        "train/grad_norm_mean": mean_grad_norm,
                        "train/clipped_grad_norm_mean": clipped_grad_norm_mean,
                        "train/noise_std": stddev,
                        "perf/step_time_sec": perf_metrics["step_time_sec"],
                        "perf/throughput_samples_per_sec": perf_metrics["throughput_samples_sec"],
                        "perf/allocated_gb": perf_metrics["memory_allocated_gb"],
                        "perf/reserved_gb": perf_metrics["memory_reserved_gb"],
                        "perf/peak_gb": perf_metrics["memory_peak_gb"],
                    }, step=global_step)

                # Console logging
                print(
                    f"Step {global_step:4d} [E{epoch + 1} S{step_idx + 1:3d}/{expected_steps_per_epoch:3d}] | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Clip: norm={current_clip_norm:.3f}, rate={clip_rate:.1%} ({num_clipped}/{len(aux.grad_norms)}) | "
                    f"GradNorm: μ={mean_grad_norm:.3f}, σ={stddev:.4f} | "
                    f"Time: {perf_metrics['step_time_sec']:.2f}s | Mem: {perf_metrics['memory_peak_gb']:.1f}GB"
                )

            # Expensive operations (eval + privacy + audit) every eval_steps
            if global_step % args.eval_steps == 0:
                current_eval_loss = eval_loss(trainable_params)
                epsilon = accounting.epsilon_at(args.target_delta)

                metrics = {
                    "eval/loss": current_eval_loss,
                    "privacy/epsilon": epsilon,
                }
                eval_msg = f"  → Eval: loss={current_eval_loss:.4f}, ε={epsilon:.3f}"

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

    if args.adaptive_clipping:
        print("\nAdaptive clipping:")
        print(f"  Initial clip norm: {args.clip_norm:.3f}")
        print(f"  Final clip norm: {clip_state.clip_norm:.3f}")
        print(
            f"  Clip norm range: [{min(clip_norms_history):.3f}, {max(clip_norms_history):.3f}]"
        )
    else:
        print("\nFixed clipping:")
        print(f"  Clip norm: {args.clip_norm:.3f}")
        print(
            f"  Average clip rate: {sum(clip_rates_history) / len(clip_rates_history):.2%}"
        )

    final_epsilon = accounting.epsilon_at(args.target_delta)
    print("\nPrivacy:")
    if use_parallel_poisson:
        print(f"  Accounting: parallel_poisson (world_size={world_size})")
    print(f"  Target: ε={args.target_epsilon:.3f}, δ={args.target_delta:.2e} (n={global_train_size})")
    print(f"  Noise multiplier: {noise_multiplier:.4f}")
    print(f"  Final ε (theoretical): {final_epsilon:.4f}")
    if args.audit:
        audit_result = run_audit(trainable_params)
        audit_eps = audit_result.epsilon_at(delta=args.target_delta)
        audit_auc = audit_result.auc()
        print(f"  Final ε (empirical):  {audit_eps:.4f}  ({audit_result.n_in} in, {audit_result.n_out} out)")
        print(f"  Audit AUC:            {audit_auc:.4f}")
        print(f"  β @ α=0.01:           {audit_result.beta_at(alpha=0.01):.4f}")
        print(f"  β @ α=0.10:           {audit_result.beta_at(alpha=0.1):.4f}")
        if use_wandb:
            wandb.log({
                "privacy/epsilon_empirical": audit_eps,
                "privacy/audit_auc": audit_auc,
            }, step=global_step)

    # Mark training complete and print profiler summary
    profiler, _ = profiler.mark("training_complete")
    summary_profiler = sync(profiler) if is_ddp else profiler
    print("\n" + summary_profiler.final_summary())
    print("\n" + summary_profiler.checkpoint_summary())

    if use_wandb:
        wandb.finish()

    if is_ddp:
        dist.barrier(device_ids=[local_rank])
        _cleanup_distributed()

    return 0


if __name__ == "__main__":
    exit(main())
