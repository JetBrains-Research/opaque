"""End-to-end DP-SGD LoRA training example for causal language models.

This example is designed as a production-style script (not a tutorial):
- clipping + noise + accounting always enabled
- adaptive clipping enabled by default
- noise multiplier calibrated from target privacy budget
- privacy and grad-norm telemetry reported every eval_steps
- optional empirical privacy auditing
"""

import argparse

import torch
import torch.nn.functional as F
import torchopt
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import TensorDataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import opaque.accounting as acc
import opaque.auditing as auditing
from opaque.accounting import calibration as cal
from opaque.accounting.accountant import Accountant
from opaque.clipping import adaptive_clipped_grad, clipped_grad
from opaque.noise import gaussian_noise
from opaque.random import key
from opaque.utils import make_functional


def compute_causal_lm_loss(logits, targets):
    """Compute causal language modeling loss."""
    if hasattr(logits, "logits"):
        logits = logits.logits

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_budgets = targets[:, 1:].contiguous()

    # Compute cross-entropy loss
    loss = F.cross_entropy(
        shift_logits.view(-1, logits.size(-1)), shift_budgets.view(-1)
    )
    return loss


def parse_args():
    """Parse command-line arguments with logical groups."""
    parser = argparse.ArgumentParser(
        description="End-to-end DP-SGD LoRA training for Causal Language Models"
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
        default=300,
        help="Number of training examples for the run",
    )
    data_group.add_argument(
        "--num_eval_samples",
        "--num_aval_samples",
        dest="num_eval_samples",
        type=int,
        default=64,
        help="Number of samples for periodic eval-loss reporting",
    )
    data_group.add_argument(
        "--max_seq_len", type=int, default=512, help="Maximum sequence length"
    )

    train_group = parser.add_argument_group("training", "Training loop settings")
    train_group.add_argument(
        "--batch_size", type=int, default=16, help="Training batch size"
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
        "--eval_steps",
        type=int,
        default=10,
        help="Log eval loss, grad norms, and privacy every N steps",
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
        default=0.15,
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
        default=1,
        help="Microbatch size passed to clipped_grad/adaptive_clipped_grad",
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
        default=0.05,
        help="Lower bound for noise calibration search",
    )
    privacy_group.add_argument(
        "--calibration_max",
        type=float,
        default=50.0,
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
        default=128,
        help="Batch size used in auditing evaluate()",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("DP-SGD LoRA Training for Causal Language Models")
    print("=" * 80)

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

    # Auto-detect eager attention for MPS
    use_eager = args.use_eager_attention or device.type == "mps"

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

    # Load model
    model_kwargs = {"config": config, "trust_remote_code": True}
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

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset, split=args.dataset_split)
    print(f"Total examples in dataset: {len(dataset)}")

    # Extract text
    all_texts = [
        item[args.dataset_text_field]
        for item in dataset.select(range(min(args.num_train_samples, len(dataset))))
    ]
    print(f"Using {len(all_texts)} training samples")

    # Tokenize
    print("Tokenizing...")
    all_encodings = tokenizer(
        all_texts,
        padding=True,
        truncation=True,
        max_length=args.max_seq_len,
        return_tensors="pt",
    )
    all_tokens = all_encodings["input_ids"].to(device)
    print(f"Token shape: {all_tokens.shape}")

    # Build train/eval tensors
    if all_tokens.size(0) < args.batch_size:
        raise ValueError(
            f"Need at least batch_size={args.batch_size} samples, got {all_tokens.size(0)}"
        )

    eval_count = min(args.num_eval_samples, all_tokens.size(0))
    eval_tokens = all_tokens[:eval_count]
    train_tokens = all_tokens

    train_dataset = TensorDataset(train_tokens, train_tokens)
    print(
        f"Prepared TensorDataset with {len(train_dataset)} train samples and {len(eval_tokens)} eval samples"
    )

    # Auditing setup at beginning of training
    audit_experiment = None
    audit_train_dataset = train_dataset
    if args.audit:
        max_canaries = max(1, len(train_dataset) - args.batch_size)
        num_canaries = min(args.audit_canaries, max_canaries)
        print(f"Setting up auditing with {num_canaries} canaries...")
        audit_experiment = auditing.setup(
            train_dataset,
            num_canaries=num_canaries,
            key=key(args.seed),
        )
        audit_train_dataset = audit_experiment.subset(train_dataset)
        print(
            f"Auditing ready: train subset size={len(audit_train_dataset)}, "
            f"in={len(audit_experiment.in_indices)}, out={len(audit_experiment.out_indices)}"
        )

    # Materialize training tensor after optional auditing split
    if args.audit and audit_experiment is not None:
        train_indices = torch.as_tensor(
            audit_experiment.train_indices(len(train_dataset)),
            dtype=torch.long,
            device=train_tokens.device,
        )
        train_tokens_for_loop = train_tokens[train_indices]
    else:
        train_tokens_for_loop = train_tokens

    num_batches = len(train_tokens_for_loop) // args.batch_size
    if num_batches == 0:
        raise ValueError(
            f"Training subset too small after auditing split: {len(train_tokens_for_loop)}"
        )

    # Convert to functional (only LoRA parameters)
    print("\nConverting to functional form (LoRA parameters only)...")
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    fmodel, trainable_params, frozen_params = make_functional(
        model,
        disable_autograd_tracking=True,
        partition_trainable=True,
    )
    param_names = list(trainable_params.keys())
    print(f"Trainable parameters: {len(param_names)}")

    def merged_params(trainable):
        return {**frozen_params, **trainable}

    # Define per-example loss
    def per_example_loss_fn(trainable, tokens_single):
        tokens_batch = tokens_single.unsqueeze(0)
        logits = fmodel(merged_params(trainable), tokens_batch)
        return compute_causal_lm_loss(logits, tokens_batch)

    def eval_loss(trainable):
        with torch.no_grad():
            logits = fmodel(merged_params(trainable), eval_tokens)
            return compute_causal_lm_loss(logits, eval_tokens).item()

    def auditing_loss_fn(trainable, x, y):
        del y
        logits = fmodel(merged_params(trainable), x.unsqueeze(0))
        return compute_causal_lm_loss(logits, x.unsqueeze(0))

    # Setup optimizer
    print("\nSetting up DP-SGD training...")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Clip norm: {args.clip_norm}")
    print(f"  Microbatch size: {args.microbatch_size}")
    print(f"  Adaptive clipping: {args.adaptive_clipping}")
    print(f"  Eval steps: {args.eval_steps}")
    print(f"  Auditing enabled: {args.audit}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Total steps: {args.num_epochs * num_batches}")

    if args.optimizer == "sgd":
        base_opt = torchopt.sgd(lr=args.learning_rate)
    else:
        base_opt = torchopt.adam(lr=args.learning_rate)

    if args.adaptive_clipping:
        grad_fn, clip_state = adaptive_clipped_grad(
            per_example_loss_fn,
            argnums=0,
            batch_argnums=(1,),
            initial_clip_norm=args.clip_norm,
            target_quantile=1.0 - args.target_clip_rate,
            clip_norm_max=args.clip_norm_max,
            microbatch_size=args.microbatch_size,
            keep_batch_dim=False,
            return_aux=True,
            key=key(args.seed),
        )
        opt_state = base_opt.init(trainable_params)
        fixed_clip_norm = None
    else:
        opt_state = base_opt.init(trainable_params)
        fixed_clip_norm = args.clip_norm
        clip_state = None

    # Noise seed for DP (automatically shifted by rank in distributed mode)
    noise_seed = args.seed

    # Pre-create clipped_grad function for fixed clipping
    if not args.adaptive_clipping:
        print("Creating clipped_grad function...")
        fixed_clipped_grad_fn, clip_state = clipped_grad(
            per_example_loss_fn,
            argnums=0,
            batch_argnums=(1,),
            l2_clip_norm=fixed_clip_norm,
            microbatch_size=args.microbatch_size,
            keep_batch_dim=False,
            return_aux=True,
        )
    else:
        fixed_clipped_grad_fn = None

    # Calibrate noise multiplier from target privacy budget
    sample_rate = args.batch_size / len(train_tokens_for_loop)
    total_steps = args.num_epochs * num_batches
    budget = cal.epsilon_budget(args.target_epsilon, delta=args.target_delta)
    calibration = cal.calibrate(
        budget,
        lambda nm: acc.poisson(acc.gaussian(nm), sample_rate=sample_rate) * total_steps,
        param_min=args.calibration_min,
        param_max=args.calibration_max,
        tolerance=args.calibration_tolerance,
    )
    noise_multiplier = calibration.param

    print("\nCalibrated privacy parameters:")
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
        epoch_losses = []

        # Build shuffled fixed-size batches from audit-aware training subset
        indices = torch.randperm(len(train_tokens_for_loop), device="cpu")
        for batch_idx in range(num_batches):
            batch_indices = indices[
                batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size
            ]
            tokens = train_tokens_for_loop[
                batch_indices.to(train_tokens_for_loop.device)
            ]

            # Determine clip norm
            current_clip_norm = (
                fixed_clip_norm if fixed_clip_norm is not None else clip_state.clip_norm
            )

            # Compute clipped gradients (with state passing)
            if fixed_clipped_grad_fn is not None:
                (grads_tuple, aux), clip_state = fixed_clipped_grad_fn(
                    trainable_params, tokens, state=clip_state
                )
            else:
                # Adaptive: grad_fn already handles adaptive clipping
                (grads_tuple, aux), clip_state = grad_fn(
                    trainable_params, tokens, state=clip_state
                )

            # Add Gaussian noise
            stddev = noise_multiplier * clip_state.sensitivity()
            noise_fn, noise_state = gaussian_noise(stddev=stddev, key=key(noise_seed))
            noisy_grads, _ = noise_fn(grads_tuple, noise_state)

            # Optimizer step (no adapter wrapper - optimizer used directly)
            updates, opt_state = base_opt.update(
                noisy_grads, opt_state, params=trainable_params
            )
            metrics = {
                "clip_norm": current_clip_norm
                if fixed_clip_norm is not None
                else clip_state.clip_norm,
                "clip_rate": clip_state.clipping_rate
                if hasattr(clip_state, "clipping_rate")
                else (aux.grad_norms > current_clip_norm).float().mean().item(),
            }

            # Apply updates
            trainable_params = torchopt.apply_updates(trainable_params, updates)

            # Accounting update
            accounting |= step_process

            # Track metrics
            avg_loss = aux.loss_values.mean().item()

            min_grad_norm = aux.grad_norms.min().item()
            max_grad_norm = aux.grad_norms.max().item()
            mean_grad_norm = aux.grad_norms.mean().item()
            median_grad_norm = aux.grad_norms.median().item()

            num_clipped = (aux.grad_norms > current_clip_norm).sum().item()

            losses.append(avg_loss)
            epoch_losses.append(avg_loss)
            clip_norms_history.append(metrics["clip_norm"])
            clip_rates_history.append(metrics["clip_rate"])

            global_step += 1

            if global_step % args.eval_steps == 0:
                epsilon = accounting.epsilon_at(args.target_delta)
                current_eval_loss = eval_loss(trainable_params)
                print(
                    f"Step {global_step:4d} [E{epoch + 1} B{batch_idx + 1:3d}/{num_batches:3d}] | "
                    f"TrainLoss: {avg_loss:.4f} | EvalLoss: {current_eval_loss:.4f} | "
                    f"ε={epsilon:.3f} (δ={args.target_delta:.1e}) | "
                    f"Clip: norm={metrics['clip_norm']:.3f}, rate={metrics['clip_rate']:.1%} ({num_clipped}/{len(aux.grad_norms)}) | "
                    f"GradNorms: μ={mean_grad_norm:.3f}, med={median_grad_norm:.3f}, "
                    f"min={min_grad_norm:.3f}, max={max_grad_norm:.3f} | "
                    f"Noise: σ={stddev:.4f}"
                )

        # Epoch summary
        epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch + 1} avg loss: {epoch_avg_loss:.4f}")

    # Final summary
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset} ({len(all_texts)} samples)")
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
        print(f"  Clip norm: {fixed_clip_norm:.3f}")
        print(
            f"  Average clip rate: {sum(clip_rates_history) / len(clip_rates_history):.2%}"
        )

    print("\nPrivacy:")
    print(f"  Target epsilon: {args.target_epsilon:.3f}")
    print(f"  Target delta: {args.target_delta:.1e}")
    print(f"  Noise multiplier (calibrated): {noise_multiplier:.4f}")
    print(f"  Final epsilon: {accounting.epsilon_at(args.target_delta):.3f}")

    if args.audit and audit_experiment is not None:
        print("\nRunning empirical privacy auditing...")
        audit_result = auditing.evaluate(
            audit_experiment,
            auditing_loss_fn,
            trainable_params,
            train_dataset,
            batch_size=args.audit_batch_size,
        )
        print(audit_result.summary(delta=args.target_delta))
        print(
            f"Theoretical ε={accounting.epsilon_at(args.target_delta):.3f} | "
            f"Empirical ε(lower bound)={audit_result.epsilon_at(delta=args.target_delta):.3f}"
        )

    return 0


if __name__ == "__main__":
    exit(main())
