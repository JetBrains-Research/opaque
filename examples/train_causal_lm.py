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
  # Configure W&B (optional):
  export WANDB_API_KEY='your-key-here'
  export WANDB_BASE_URL='https://jetbrains.wandb.io'  # For JetBrains W&B instance
  export WANDB_ENTITY='federated-compute'  # Your team/entity name

  python examples/train_causal_lm.py --preset mellum-kstack --wandb

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
    --lora_budget_modules q_proj k_proj v_proj o_proj \\
    --audit --audit_canaries 1000 \\
    --wandb
"""

import argparse
import warnings

import torch
import torchopt
from datasets import load_dataset, Dataset
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
from opaque.noise import gaussian_noise
from opaque.random import key, fold_in
from opaque.sampling import PoissonSampler
from opaque.utils import make_functional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_args():
    """Parse command-line arguments with logical groups."""
    parser = argparse.ArgumentParser(
        description="End-to-end DP-SGD LoRA training for Causal Language Models"
    )

    # Preset configurations
    parser.add_argument(
        "--preset",
        type=str,
        choices=["smoke", "mellum-kstack"],
        default="smoke",
        help="Apply preset configuration (smoke=quick test ~2min, medium=longer test ~10min, mellum-kstack=full production). Overrides other args.",
    )

    model_group = parser.add_argument_group("model", "Model and tokenizer settings")
    model_group.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="HuggingFace model name or local path",
    )
    model_group.add_argument(
        "--use_eager_attention",
        action="store_true",
        help="Force eager attention implementation",
    )

    data_group = parser.add_argument_group("data", "Dataset and tokenization settings")
    data_group.add_argument(
        "--dataset", type=str, default="ag_news", help="HuggingFace dataset name"
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
    train_group.add_argument("--seed", type=int, default=42, help="Random seed")

    lora_group = parser.add_argument_group("lora", "LoRA adapter settings")
    lora_group.add_argument("--lora_r", type=int, default=4, help="LoRA rank")
    lora_group.add_argument("--lora_alpha", type=int, default=8, help="LoRA alpha")
    lora_group.add_argument(
        "--lora_budget_modules",
        type=str,
        nargs="+",
        default=["c_attn", "c_proj"],
        help="Target module names for LoRA",
    )

    dp_group = parser.add_argument_group("dp", "DP-SGD clipping and noise")
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
        default=1e-5,
        help="Target delta used in accounting and calibration",
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
        default=100,
        help="Number of canaries for one-run auditing",
    )
    audit_group.add_argument(
        "--audit_batch_size",
        type=int,
        default=32,
        help="Batch size used in auditing and eval",
    )

    wandb_group = parser.add_argument_group("wandb", "Weights & Biases tracking")
    wandb_group.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases tracking (requires WANDB_API_KEY env var)",
    )
    wandb_group.add_argument(
        "--wandb_project",
        type=str,
        default="opaque",
        help="W&B project name (default: opaque)",
    )
    wandb_group.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity/team name (default: read from WANDB_ENTITY env var or use personal account)",
    )
    wandb_group.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name (default: auto-generated from model and hyperparameters)",
    )

    args = parser.parse_args()

    # Apply preset configurations
    if args.preset == "smoke":
        # Quick smoke test with GPT-2 (~100 steps, ~2-3 minutes)
        args.model_name = "gpt2"
        args.dataset = "ag_news"
        args.dataset_text_field = "text"
        args.num_train_samples = 1000
        args.num_eval_samples = 100
        args.num_epochs = 3
        args.batch_size = 32
        args.eval_batch_size = 8  # Small eval batches
        args.log_steps = 10
        args.eval_steps = 10
        args.target_epsilon = 3.0
        args.learning_rate = 1e-5
        args.lora_r = 4
        args.lora_alpha = 8
        args.max_seq_len = 512
        args.lora_budget_modules = ["c_attn", "c_proj"]
        args.use_eager_attention = True  # Required: SDPA incompatible with vmap
        args.dtype = "bfloat16"  # Use bfloat16 by default
        args.audit = False
    elif args.preset == "mellum-kstack":
        # Golden configuration for Mellum-4b + KStack training on H200
        # Memory analysis: Model=7.5 GiB, Activations per example=~17 GiB (bfloat16, seq_len=1024)
        # With microbatch_size=4: 7.5 + (4×17) = ~75 GiB peak memory usage
        args.model_name = "JetBrains/Mellum-4b-base"
        args.dataset = "JetBrains/KStack"
        args.dataset_text_field = "content"
        args.num_train_samples = 50000
        args.num_eval_samples = 1000
        args.num_epochs = 3
        args.batch_size = 128  # Large batch for better privacy amplification
        args.eval_batch_size = 4  # Small batches for eval to avoid OOM
        args.log_steps = 2  # Frequent logging
        args.eval_steps = 10  # Regular evaluation
        args.target_epsilon = 10.0
        args.learning_rate = 5e-5
        args.lora_r = 16
        args.lora_alpha = 32
        args.max_seq_len = 1024
        args.lora_budget_modules = ["q_proj", "v_proj"]  # Minimal LoRA for memory efficiency
        args.use_eager_attention = True  # Required: SDPA incompatible with vmap
        args.dtype = "bfloat16"  # Required: Cuts memory by ~50% vs FP32
        args.microbatch_size = 4  # Required: Process 4 examples at a time (vmap limitation)

    return args


def main():
    args = parse_args()

    # Set eval_batch_size to batch_size if not specified
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size

    print("=" * 80)
    print("DP-SGD LoRA Training for Causal Language Models")
    print("=" * 80)

    # Initialize W&B if enabled
    use_wandb = args.wandb and WANDB_AVAILABLE
    if use_wandb:
        import os

        # Read entity from env var if not specified
        entity = args.wandb_entity or os.environ.get("WANDB_ENTITY")

        # Generate default run name from key parameters if not specified
        if args.wandb_run_name is None:
            model_short = args.model_name.split('/')[-1]
            run_name = f"{model_short}_n{args.num_train_samples}_e{args.num_epochs}_b{args.batch_size}_eps{args.target_epsilon}_lr{args.learning_rate}"
        else:
            run_name = args.wandb_run_name

        wandb.init(
            project=args.wandb_project,
            entity=entity,
            name=run_name,
            config=vars(args),
        )
        print(f"W&B initialized: {wandb.run.url}")
    elif args.wandb and not WANDB_AVAILABLE:
        print("Warning: --wandb specified but wandb not installed. Continuing without W&B.")

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"\nUsing device: {device} ({device_name})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\nUsing device: {device} (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print(f"\nUsing device: {device}")
        print("Warning: Training on CPU will be slow")

    # Set seed
    torch.manual_seed(args.seed)

    # Force eager attention for DP-SGD training
    # SDPA (scaled_dot_product_attention) has compatibility issues with vmap for some models
    # TODO: Re-enable SDPA support after fixing vmap compatibility issues
    use_eager = True
    if not args.use_eager_attention:
        print("Auto-enabling eager attention (required for DP-SGD with vmap)")

    # Legacy checks (kept for reference, but now always using eager)
    # use_eager = args.use_eager_attention or device.type == "mps" or (args.microbatch_size is not None and args.microbatch_size > 1)

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

    # Map dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]
    print(f"Using dtype: {args.dtype} ({torch_dtype})")

    # Load model
    model_kwargs = {
        "config": config,
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }
    if use_eager:
        print("Using eager attention implementation")
        model_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model = model.to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_budget_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and prepare dataset
    print(f"\nLoading dataset: {args.dataset}...")
    print(f"  Split: {args.dataset_split}")
    print(f"  Text field: {args.dataset_text_field}")
    dataset = load_dataset(args.dataset, split=args.dataset_split)
    print(f"  Total examples in dataset: {len(dataset)}")

    # Validate we have enough data
    total_needed = args.num_train_samples + args.num_eval_samples
    if len(dataset) < total_needed:
        raise ValueError(
            f"Dataset has {len(dataset)} examples but need {total_needed} "
            f"(train={args.num_train_samples} + eval={args.num_eval_samples})"
        )

    # Show sample of raw data
    if len(dataset) > 0:
        sample = dataset[0]
        sample_text = sample[args.dataset_text_field]
        print(f"\n  Sample data (first example):")
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

    # Eval DataLoader (standard batching, no privacy requirements)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
        drop_last=False,
    )

    # For training: Poisson sampling (not uniform shuffling!)
    # Poisson: each example independently sampled with probability sample_rate each step
    sample_rate = args.batch_size / len(train_dataset)

    # Expected number of steps to process full dataset with Poisson sampling
    # = 1 / sample_rate (since we sample sample_rate fraction each step)
    expected_steps_per_epoch = int(1.0 / sample_rate)

    print(f"\nPoisson sampling setup:")
    print(f"  Sample rate: {sample_rate:.4f}")
    print(f"  Expected batch size: {args.batch_size}")
    print(f"  Expected steps per epoch: ~{expected_steps_per_epoch}")
    print(f"Eval batches: {len(eval_loader)}")

    # Convert to functional (only LoRA parameters)
    print("\nConverting to functional form (LoRA parameters only)...")
    print("  (This may take 1-2 minutes for large models...)")
    import time
    start_time = time.time()
    fmodel, trainable_params, frozen_params = make_functional(
        model,
        disable_autograd_tracking=True,
        partition_trainable=True,
    )
    param_names = list(trainable_params.keys())
    elapsed = time.time() - start_time
    print(f"Trainable parameters: {len(param_names)} (took {elapsed:.1f}s)")

    def merged_params(trainable):
        return {**frozen_params, **trainable}

    # Define per-example loss
    def per_example_loss_fn(trainable, tokens_batch):
        # tokens_batch has shape (1, seq_len) when keep_batch_dim=True
        # Use HuggingFace's built-in loss (handles shifting internally)
        output = fmodel(merged_params(trainable), tokens_batch, labels=tokens_batch)
        return output.loss

    def eval_loss(trainable):
        """Compute eval loss using DataLoader."""
        with torch.no_grad():
            total_loss = 0.0
            total_tokens = 0

            for batch in eval_loader:
                batch_tokens = batch["input_ids"].to(device)
                output = fmodel(merged_params(trainable), batch_tokens, labels=batch_tokens)
                total_loss += output.loss.item() * len(batch_tokens)
                total_tokens += len(batch_tokens)

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
            keep_batch_dim=True,
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
            keep_batch_dim=True,
            return_aux=True,
        )

    opt_state = base_opt.init(trainable_params)

    # Calibrate noise multiplier from target privacy budget
    # sample_rate already computed above
    total_steps = args.num_epochs * expected_steps_per_epoch
    print(f"\nCalibrating privacy parameters...")
    print(f"  Total steps: {total_steps}")
    print(f"  Sample rate: {sample_rate:.6f}")
    print(f"  Target: ε={args.target_epsilon}, δ={args.target_delta}")
    print(f"  (This may take 1-3 minutes...)")

    start_time = time.time()
    budget = cal.epsilon_budget(args.target_epsilon, delta=args.target_delta)
    calibration = cal.calibrate(
        budget,
        lambda nm: acc.poisson(acc.gaussian(nm), sample_rate=sample_rate) * total_steps,
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

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 80)
        print("Creating Poisson sampler...")

        # Create Poisson sampler for this epoch
        epoch_sampler = PoissonSampler(
            train_dataset,
            sample_rate=sample_rate,
            num_iterations=expected_steps_per_epoch,
            key=fold_in(key(args.seed), epoch),
        )
        print("Creating DataLoader...")

        # DataLoader with batch_sampler
        epoch_loader = DataLoader(
            train_dataset,
            batch_sampler=epoch_sampler,
            collate_fn=data_collator,
        )

        # Iterate through Poisson-sampled batches
        for step_idx, batch in enumerate(epoch_loader):
            tokens = batch["input_ids"].to(device)

            # Accounting update (must happen even for empty batches)
            accounting |= step_process

            # Skip if no examples sampled (rare but possible with Poisson)
            if len(tokens) == 0:
                continue

            # Compute clipped gradients (with state passing)
            (grads_tuple, aux), clip_state = grad_fn(
                trainable_params, tokens, state=clip_state
            )
            current_clip_norm = clip_state.clip_norm

            # Add Gaussian noise
            stddev = noise_multiplier * clip_state.sensitivity()
            noisy_grads, noise_state = noise_fn(grads_tuple, noise_state)

            # Optimizer step (no adapter wrapper - optimizer used directly)
            updates, opt_state = base_opt.update(
                noisy_grads, opt_state, params=trainable_params
            )

            # Apply updates
            trainable_params = torchopt.apply_updates(trainable_params, updates)

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

                # W&B logging
                if use_wandb:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/clip_norm": current_clip_norm,
                        "train/clip_rate": clip_rate,
                        "train/grad_norm_mean": mean_grad_norm,
                        "train/clipped_grad_norm_mean": clipped_grad_norm_mean,
                        "train/noise_std": stddev,
                        "train/step": global_step,
                    }, step=global_step)

                # Console logging
                print(
                    f"Step {global_step:4d} [E{epoch + 1} S{step_idx + 1:3d}/{expected_steps_per_epoch:3d}] | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Clip: norm={current_clip_norm:.3f}, rate={clip_rate:.1%} ({num_clipped}/{len(aux.grad_norms)}) | "
                    f"GradNorm: μ={mean_grad_norm:.3f}, σ={stddev:.4f}"
                )

            # Expensive operations (eval + privacy) every eval_steps
            if global_step % args.eval_steps == 0:
                epsilon = accounting.epsilon_at(args.target_delta)
                current_eval_loss = eval_loss(trainable_params)

                # W&B logging - eval metrics
                if use_wandb:
                    wandb.log({
                        "eval/loss": current_eval_loss,
                        "privacy/epsilon": epsilon,
                        "privacy/delta": args.target_delta,
                    }, step=global_step)

                print(
                    f"  → Eval: loss={current_eval_loss:.4f}, ε={epsilon:.3f} (δ={args.target_delta:.1e})"
                )

    # Final summary
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset} ({len(train_dataset)} train samples)")
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

    print("\nPrivacy:")
    print(f"  Target epsilon: {args.target_epsilon:.3f}")
    print(f"  Target delta: {args.target_delta:.1e}")
    print(f"  Noise multiplier (calibrated): {noise_multiplier:.4f}")
    print(f"  Final epsilon: {accounting.epsilon_at(args.target_delta):.3f}")

    return 0


if __name__ == "__main__":
    exit(main())
