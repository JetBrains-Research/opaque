"""Universal DP-SGD LoRA fine-tuning for Causal Language Models.

Supports any HuggingFace causal LM with configurable hyperparameters for:
- Model selection (Qwen, LLaMA, GPT-2, etc.)
- Dataset selection (AG News, IMDB, custom text datasets)
- Training hyperparameters (batch size, learning rate, epochs, etc.)
- LoRA configuration (rank, alpha, target modules)
- DP-SGD parameters (clip norm, noise multiplier, adaptive clipping)
"""

import argparse

import torch
import torch.nn.functional as F
import torchopt
from datasets import load_dataset
from opaque.optimizers import adaptive_clipping
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from opaque.clipping import clipped_grad
from opaque.noise import add_gaussian_noise
from opaque.utils import make_functional


def compute_causal_lm_loss(logits, targets):
    """Compute causal language modeling loss."""
    if hasattr(logits, "logits"):
        logits = logits.logits

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()

    # Compute cross-entropy loss
    loss = F.cross_entropy(
        shift_logits.view(-1, logits.size(-1)), shift_targets.view(-1)
    )
    return loss


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="DP-SGD LoRA training for Causal Language Models"
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--use_eager_attention", action="store_true", help="Use eager attention"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset", type=str, default="ag_news", help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--dataset_split", type=str, default="train", help="Dataset split to use"
    )
    parser.add_argument(
        "--dataset_text_field",
        type=str,
        default="text",
        help="Field containing text data",
    )
    parser.add_argument(
        "--num_train_samples",
        type=int,
        default=20000,
        help="Number of training samples to use",
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=1024, help="Maximum sequence length"
    )

    # Training arguments
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1.0e-5, help="Learning rate"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["sgd", "adam"],
        help="Optimizer to use",
    )
    parser.add_argument("--seed", type=int, default=56, help="Random seed")

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument(
        "--lora_alpha", type=int, default=32, help="LoRA alpha (scaling factor)"
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        help="Target modules for LoRA",
    )

    # DP-SGD arguments
    parser.add_argument(
        "--clip_norm", type=float, default=0.15, help="Gradient clipping norm"
    )
    parser.add_argument(
        "--noise_multiplier", type=float, default=0.24, help="Noise multiplier for DP"
    )
    parser.add_argument(
        "--microbatch_size",
        type=int,
        default=4,
        help="Microbatch size for gradient computation",
    )
    parser.add_argument(
        "--use_adaptive_clipping",
        action="store_true",
        default=True,
        help="Use adaptive clipping (adapts clip norm during training)",
    )
    parser.add_argument(
        "--no_adaptive_clipping",
        dest="use_adaptive_clipping",
        action="store_false",
        help="Disable adaptive clipping (use fixed clip norm)",
    )
    parser.add_argument(
        "--target_clip_rate",
        type=float,
        default=0.50,
        help="Target clipping rate for adaptive clipping (target_unclipped_quantile)",
    )
    parser.add_argument(
        "--clip_norm_max",
        type=float,
        default=10.0,
        help="Maximum clip norm for adaptive clipping",
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
        target_modules=args.lora_target_modules,
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

    # Create batches
    num_batches = len(all_texts) // args.batch_size
    batches = [
        all_tokens[i * args.batch_size : (i + 1) * args.batch_size]
        for i in range(num_batches)
    ]
    print(f"Created {len(batches)} batches of size {args.batch_size}")

    # Convert to functional (only LoRA parameters)
    print("\nConverting to functional form (LoRA parameters only)...")
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    fmodel, params = make_functional(model, disable_autograd_tracking=True)
    param_names = [
        name for name, param in model.named_parameters() if param.requires_grad
    ]
    print(f"Trainable parameters: {len(param_names)}")

    # Define per-example loss
    def per_example_loss_fn(params_tuple, tokens_single):
        tokens_batch = tokens_single.unsqueeze(0)
        logits = fmodel(params_tuple, tokens_batch)
        return compute_causal_lm_loss(logits, tokens_batch)

    # Setup optimizer
    print("\nSetting up DP-SGD training...")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Clip norm: {args.clip_norm}")
    print(f"  Noise multiplier: {args.noise_multiplier}")
    print(f"  Microbatch size: {args.microbatch_size}")
    print(f"  Adaptive clipping: {args.use_adaptive_clipping}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Total steps: {args.num_epochs * len(batches)}")

    if args.optimizer == "sgd":
        base_opt = torchopt.sgd(lr=args.learning_rate)
    else:
        base_opt = torchopt.adam(lr=args.learning_rate)

    if args.use_adaptive_clipping:
        init_fn, step_fn = adaptive_clipping(
            base_opt,
            initial_clip_norm=args.clip_norm,
            target_clip_rate=args.target_clip_rate,
            clip_norm_max=args.clip_norm_max,
        )
        opt_state = init_fn(params)
        fixed_clip_norm = None
    else:
        opt_state = base_opt.init(params)
        fixed_clip_norm = args.clip_norm

    # Setup RNG
    if device.type == "cpu":
        rng = torch.Generator().manual_seed(args.seed)
    else:
        rng = torch.Generator(device=device).manual_seed(args.seed)

    # Pre-create clipped_grad function for fixed clipping
    if not args.use_adaptive_clipping:
        print("Creating clipped_grad function...")
        fixed_clipped_grad_fn = clipped_grad(
            per_example_loss_fn,
            argnums=0,
            batch_argnums=(1,),
            l2_clip_norm=fixed_clip_norm,
            microbatch_size=args.microbatch_size,
            keep_batch_dim=False,
            return_grad_norms=True,
            return_values=True,
        )
    else:
        fixed_clipped_grad_fn = None

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

        for batch_idx, tokens in enumerate(batches):
            # Determine clip norm
            current_clip_norm = (
                fixed_clip_norm
                if fixed_clip_norm is not None
                else opt_state.current_clip_norm
            )

            # Compute clipped gradients
            if fixed_clipped_grad_fn is not None:
                grads_tuple, aux = fixed_clipped_grad_fn(params, tokens)
            else:
                clipped_grad_fn = clipped_grad(
                    per_example_loss_fn,
                    argnums=0,
                    batch_argnums=(1,),
                    l2_clip_norm=current_clip_norm,
                    microbatch_size=args.microbatch_size,
                    keep_batch_dim=False,
                    return_grad_norms=True,
                    return_values=True,
                )
                grads_tuple, aux = clipped_grad_fn(params, tokens)

            # Add Gaussian noise
            stddev = args.noise_multiplier * current_clip_norm
            noisy_grads = add_gaussian_noise(grads_tuple, stddev=stddev, generator=rng)

            # Optimizer step
            if args.use_adaptive_clipping:
                updates, opt_state, metrics = step_fn(
                    noisy_grads, aux.grad_norms, opt_state, params=params
                )
            else:
                updates, opt_state = base_opt.update(
                    noisy_grads, opt_state, params=params
                )
                clip_rate = (aux.grad_norms > current_clip_norm).float().mean().item()
                metrics = {
                    "clip_norm": current_clip_norm,
                    "clip_rate": clip_rate,
                }

            # Apply updates
            params = torchopt.apply_updates(params, updates)

            # Track metrics
            avg_loss = aux.values.mean().item()
            min_loss = aux.values.min().item()
            max_loss = aux.values.max().item()
            loss_std = aux.values.std().item()

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

            # Print detailed telemetry every step
            print(
                f"Step {global_step:4d} [E{epoch + 1} B{batch_idx + 1:3d}/{len(batches):3d}] | "
                f"Loss: {avg_loss:.4f} (min={min_loss:.4f}, max={max_loss:.4f}, std={loss_std:.4f}) | "
                f"Clip: norm={metrics['clip_norm']:.3f}, rate={metrics['clip_rate']:.1%} ({num_clipped}/{len(aux.grad_norms)}) | "
                f"Grad: μ={mean_grad_norm:.3f}, med={median_grad_norm:.3f}, σ=[{min_grad_norm:.3f}, {max_grad_norm:.3f}] | "
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

    if args.use_adaptive_clipping:
        print("\nAdaptive clipping:")
        print(f"  Initial clip norm: {args.clip_norm:.3f}")
        print(f"  Final clip norm: {clip_norms_history[-1]:.3f}")
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
    print(f"  Noise multiplier: {args.noise_multiplier}")

    return 0


if __name__ == "__main__":
    exit(main())
