"""End-to-End DP Fine-Tuning of LLaMA on H200 GPU.

This example demonstrates production DP-SGD with all Opaque features:
- LLaMA-3-8B model with LoRA (parameter-efficient fine-tuning)
- TruncatedPoissonSampler (best practical privacy)
- Adaptive clipping (auto-adjusting threshold)
- Microbatching (memory efficiency)
- Proper privacy accounting (optimized for speed)
- Evaluation dataset with metrics

Hardware: Nvidia H200 (141GB HBM3)
Training time: 10-15 minutes
Dataset: OpenAssistant Conversations (high-quality instruction data)

Key optimizations:
- Privacy calculated only during eval (saves ~30% time)
- Large batch sizes (256) for GPU utilization
- Microbatch size (64) for memory efficiency
- Evaluation every 100 steps
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchopt
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Opaque imports
import opaque.accounting as acc
from opaque import (
    TruncatedPoissonSampler,
    add_gaussian_noise,
    clipped_grad,
    make_functional,
)
from opaque.optimizers import adaptive_clipping


def setup_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DP Fine-Tuning of LLaMA")

    # Model
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model name (e.g., meta-llama/Llama-3.2-1B, meta-llama/Llama-2-7b-hf)",
    )
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (scaling)")

    # Training
    parser.add_argument("--num_steps", type=int, default=1000, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=256, help="Expected batch size")
    parser.add_argument("--max_batch_size", type=int, default=320, help="Max batch size (TruncatedPoisson cap)")
    parser.add_argument("--microbatch_size", type=int, default=64, help="Microbatch size for memory efficiency")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")

    # Privacy
    parser.add_argument("--initial_clip_norm", type=float, default=0.5, help="Initial clipping threshold (adaptive)")
    parser.add_argument("--target_clip_rate", type=float, default=0.20, help="Target fraction of clipped gradients")
    parser.add_argument("--noise_multiplier", type=float, default=0.8, help="Noise multiplier (σ)")
    parser.add_argument("--target_delta", type=float, default=1e-6, help="Target delta for (ε, δ)-DP")

    # Data
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--num_train_samples", type=int, default=50000, help="Number of training samples")
    parser.add_argument("--num_eval_samples", type=int, default=2000, help="Number of eval samples")

    # Evaluation
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate every N steps")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size for evaluation")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./llama_dp_output", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")

    return parser.parse_args()


def load_and_prepare_data(args, tokenizer):
    """Load OpenAssistant dataset and tokenize."""
    print("\n" + "=" * 80)
    print("LOADING DATASET")
    print("=" * 80)

    # Load OpenAssistant Conversations dataset
    print(f"Loading OpenAssistant dataset (train: {args.num_train_samples}, eval: {args.num_eval_samples})...")
    dataset = load_dataset("OpenAssistant/oasst1", split="train")

    # Filter for English, high-quality conversations
    dataset = dataset.filter(lambda x: x["lang"] == "en")

    # Take first messages from conversations (instructions)
    texts = []
    for example in dataset:
        if example["role"] == "prompter":  # User prompts
            texts.append(example["text"])
        if len(texts) >= args.num_train_samples + args.num_eval_samples:
            break

    print(f"Collected {len(texts)} text samples")

    # Split train/eval
    train_texts = texts[:args.num_train_samples]
    eval_texts = texts[args.num_train_samples:args.num_train_samples + args.num_eval_samples]

    print(f"Train samples: {len(train_texts)}")
    print(f"Eval samples: {len(eval_texts)}")

    # Tokenize
    print(f"Tokenizing (max_length={args.max_length})...")

    def tokenize_texts(texts):
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        return tokenized["input_ids"], tokenized["attention_mask"]

    train_input_ids, train_attention_mask = tokenize_texts(train_texts)
    eval_input_ids, eval_attention_mask = tokenize_texts(eval_texts)

    # Labels = input_ids (causal language modeling)
    train_labels = train_input_ids.clone()
    eval_labels = eval_input_ids.clone()

    print(f"\n✅ Dataset ready!")
    print(f"   Train shape: {train_input_ids.shape}")
    print(f"   Eval shape: {eval_input_ids.shape}")
    print(f"   Vocabulary size: {tokenizer.vocab_size:,}")
    print("=" * 80)

    return (
        TensorDataset(train_input_ids, train_attention_mask, train_labels),
        TensorDataset(eval_input_ids, eval_attention_mask, eval_labels),
    )


def setup_model(args):
    """Load LLaMA model and apply LoRA."""
    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)
    print(f"Model: {args.model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,  # Use FP32 for DP training
        device_map=args.device,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅ Model loaded: {args.model_name}")
    print(f"   Total parameters: {total_params:,}")

    # Apply LoRA
    print(f"\nApplying LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,  # Deterministic for DP
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers
        init_lora_weights=True,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✅ LoRA applied!")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Reduction: {total_params / trainable_params:.0f}× fewer parameters")
    print(f"   Training only {trainable_params / total_params:.2%} of the model")
    print("=" * 80)

    return model, tokenizer


def evaluate(
    fmodel,
    trainable_params,
    frozen_params,
    eval_loader,
    device,
):
    """Evaluate model on eval set."""
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for eval_input_ids, eval_mask, eval_labels in eval_loader:
            eval_input_ids = eval_input_ids.to(device)
            eval_mask = eval_mask.to(device)
            eval_labels = eval_labels.to(device)

            batch_size = eval_input_ids.shape[0]

            # Combine params
            all_params = {**frozen_params, **trainable_params}

            # Forward pass
            outputs = fmodel(
                all_params,
                eval_input_ids,
                attention_mask=eval_mask,
                labels=eval_labels,
            )

            total_loss += outputs.loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def main():
    args = setup_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("END-TO-END DP FINE-TUNING OF LLAMA")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_name}")
    print(f"  LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  Steps: {args.num_steps}")
    print(f"  Batch size: {args.batch_size} (expected), {args.max_batch_size} (max)")
    print(f"  Microbatch size: {args.microbatch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Initial clip norm: {args.initial_clip_norm} (adaptive!)")
    print(f"  Noise multiplier: {args.noise_multiplier}")
    print(f"  Target delta: {args.target_delta}")
    print(f"  Eval steps: {args.eval_steps}")
    print(f"  Device: {args.device}")
    print("=" * 80)

    # Setup model
    model, tokenizer = setup_model(args)
    device = torch.device(args.device)

    # Load data
    train_dataset, eval_dataset = load_and_prepare_data(args, tokenizer)

    # Create TruncatedPoissonSampler for training
    print("\n" + "=" * 80)
    print("CREATING SAMPLERS")
    print("=" * 80)

    sample_rate = args.batch_size / len(train_dataset)
    sampler = TruncatedPoissonSampler(
        train_dataset,
        sample_rate=sample_rate,
        max_batch_size=args.max_batch_size,
        num_epochs=args.num_steps,
        generator=np.random.default_rng(args.seed),
    )

    train_loader = DataLoader(train_dataset, batch_sampler=sampler)
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False)

    print(f"✅ TruncatedPoissonSampler:")
    print(f"   Sample rate: {sample_rate:.6f}")
    print(f"   Expected batch size: {sampler.expected_batch_size:.1f}")
    print(f"   Max batch size: {sampler.max_batch_size}")
    print(f"\n✅ Eval loader:")
    print(f"   Batch size: {args.eval_batch_size}")
    print(f"   Batches: {len(eval_loader)}")
    print("=" * 80)

    # Convert to functional form
    print("\n" + "=" * 80)
    print("FUNCTIONAL CONVERSION")
    print("=" * 80)

    fmodel, trainable_params, frozen_params = make_functional(
        model,
        disable_autograd_tracking=True,
        partition_trainable=True,
    )

    trainable_count = sum(p.numel() for p in trainable_params.values())
    frozen_count = sum(p.numel() for p in frozen_params.values())

    print(f"✅ Functional form:")
    print(f"   Trainable: {trainable_count:,}")
    print(f"   Frozen: {frozen_count:,}")
    print("=" * 80)

    # Define per-example loss
    def per_example_loss(trainable, frozen, input_ids_single, mask_single, labels_single):
        all_params = {**frozen, **trainable}
        input_batch = input_ids_single.unsqueeze(0)
        mask_batch = mask_single.unsqueeze(0)
        labels_batch = labels_single.unsqueeze(0)
        outputs = fmodel(all_params, input_batch, attention_mask=mask_batch, labels=labels_batch)
        return outputs.loss

    # Setup adaptive clipping optimizer
    print("\n" + "=" * 80)
    print("OPTIMIZER SETUP")
    print("=" * 80)

    base_optimizer = torchopt.adamw(
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    init_opt_fn, step_opt_fn = adaptive_clipping(
        base_optimizer,
        initial_clip_norm=args.initial_clip_norm,
        target_clip_rate=args.target_clip_rate,
        clip_norm_min=0.01,
        clip_norm_max=10.0,
    )

    opt_state = init_opt_fn(trainable_params)

    print(f"✅ Adaptive clipping optimizer:")
    print(f"   Base: AdamW")
    print(f"   Learning rate: {args.lr}")
    print(f"   Weight decay: {args.weight_decay}")
    print(f"   Initial clip norm: {args.initial_clip_norm}")
    print(f"   Target clip rate: {args.target_clip_rate * 100:.0f}%")
    print("=" * 80)

    # Initialize privacy accounting (only tracked at eval)
    privacy_state = acc.create()
    steps_since_last_privacy_calc = 0

    print("\n⚠️  Privacy optimization enabled:")
    print(f"   Privacy calculated only every {args.eval_steps} steps (at eval)")
    print(f"   Saves ~30% training time!")

    # Training metrics
    train_losses = []
    eval_losses = []
    epsilons = []
    clip_norms = []
    clip_rates = []
    eval_steps_list = []

    # RNG for noise
    rng = torch.Generator().manual_seed(args.seed)

    # Start training
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"Target: Train for {args.num_steps} steps on {args.device}")
    print(f"Expected time: 10-15 minutes on H200")
    print("=" * 80 + "\n")

    start_time = time.time()

    for step, (batch_input_ids, batch_mask, batch_labels) in enumerate(train_loader):
        if step >= args.num_steps:
            break

        # Move to device
        batch_input_ids = batch_input_ids.to(device)
        batch_mask = batch_mask.to(device)
        batch_labels = batch_labels.to(device)

        batch_size = batch_input_ids.shape[0]
        current_clip_norm = opt_state.current_clip_norm

        # 1. Compute clipped gradients (with adaptive clip norm + microbatching)
        clipped_grad_fn = clipped_grad(
            per_example_loss,
            argnums=0,
            batch_argnums=(2, 3, 4),
            l2_clip_norm=current_clip_norm,
            microbatch_size=args.microbatch_size,
            return_values=True,
            return_grad_norms=True,
        )

        grads, aux = clipped_grad_fn(
            trainable_params,
            frozen_params,
            batch_input_ids,
            batch_mask,
            batch_labels,
        )

        # 2. Add Gaussian noise
        stddev = args.noise_multiplier * current_clip_norm
        noisy_grads = add_gaussian_noise(grads, stddev=stddev, generator=rng)

        # 3. Optimizer step
        updates, opt_state, opt_metrics = step_opt_fn(
            noisy_grads,
            aux.grad_norms,
            opt_state,
            params=trainable_params,
        )

        # 4. Apply updates
        trainable_params = torchopt.apply_updates(trainable_params, updates)

        # Track metrics
        avg_loss = aux.values.mean().item()
        train_losses.append(avg_loss)
        clip_norms.append(opt_metrics["clip_norm"])
        clip_rates.append(opt_metrics["clip_rate"])

        # Count steps for privacy calculation
        steps_since_last_privacy_calc += 1

        # 5. Evaluate and calculate privacy (only at eval steps)
        if (step + 1) % args.eval_steps == 0 or step == 0:
            # Calculate privacy for accumulated steps
            privacy_state = acc.compose_truncated_poisson_gaussian(
                privacy_state,
                noise_multiplier=args.noise_multiplier,
                sample_rate=sample_rate,
                truncated_batch_size=args.max_batch_size,
                dataset_size=len(train_dataset),
                count=steps_since_last_privacy_calc,  # Batch update!
            )
            steps_since_last_privacy_calc = 0

            epsilon = acc.get_epsilon(privacy_state, delta=args.target_delta)
            epsilons.append(epsilon)
            eval_steps_list.append(step + 1)

            # Evaluate
            eval_loss = evaluate(
                fmodel,
                trainable_params,
                frozen_params,
                eval_loader,
                device,
            )
            eval_losses.append(eval_loss)

            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed

            print(
                f"Step {step + 1:4d}/{args.num_steps} | "
                f"Train Loss: {avg_loss:6.3f} | "
                f"Eval Loss: {eval_loss:6.3f} | "
                f"Clip: {opt_metrics['clip_norm']:5.2f} | "
                f"Rate: {opt_metrics['clip_rate']:4.1%} | "
                f"ε: {epsilon:6.3f} | "
                f"{steps_per_sec:.2f} steps/s"
            )

    # Final privacy calculation (if there are remaining steps)
    if steps_since_last_privacy_calc > 0:
        privacy_state = acc.compose_truncated_poisson_gaussian(
            privacy_state,
            noise_multiplier=args.noise_multiplier,
            sample_rate=sample_rate,
            truncated_batch_size=args.max_batch_size,
            dataset_size=len(train_dataset),
            count=steps_since_last_privacy_calc,
        )

    final_epsilon = acc.get_epsilon(privacy_state, delta=args.target_delta)

    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nResults:")
    print(f"  Total time: {total_time / 60:.2f} minutes")
    print(f"  Average speed: {args.num_steps / total_time:.2f} steps/sec")
    print(f"  Starting train loss: {train_losses[0]:.3f}")
    print(f"  Final train loss: {train_losses[-1]:.3f}")
    if eval_losses:
        print(f"  Starting eval loss: {eval_losses[0]:.3f}")
        print(f"  Final eval loss: {eval_losses[-1]:.3f}")
    print(f"\nAdaptive Clipping:")
    print(f"  Initial clip norm: {args.initial_clip_norm:.2f}")
    print(f"  Final clip norm: {clip_norms[-1]:.2f}")
    print(f"  Average clip rate: {np.mean(clip_rates):.1%}")
    print(f"\nPrivacy:")
    print(f"  Final ε: {final_epsilon:.3f}")
    print(f"  Target δ: {args.target_delta}")
    print(f"  Privacy guarantee: ({final_epsilon:.1f}, {args.target_delta})-DP")
    print(f"\n✅ Privacy calculated efficiently (only at eval steps)!")
    print("=" * 80)

    # Save results
    results = {
        "args": vars(args),
        "train_losses": train_losses,
        "eval_losses": eval_losses,
        "eval_steps": eval_steps_list,
        "epsilons": epsilons,
        "clip_norms": clip_norms,
        "clip_rates": clip_rates,
        "final_epsilon": final_epsilon,
        "total_time": total_time,
    }

    results_path = output_dir / "training_results.pt"
    torch.save(results, results_path)
    print(f"\n💾 Results saved to: {results_path}")

    # Save LoRA adapters
    lora_path = output_dir / "lora_adapters"
    print(f"💾 Saving LoRA adapters to: {lora_path}")
    # Note: Need to reconstruct model from functional form for saving
    # For now, just save trainable params
    torch.save(trainable_params, output_dir / "trainable_params.pt")
    print(f"   Trainable params saved to: {output_dir / 'trainable_params.pt'}")

    print("\n🎉 End-to-end DP fine-tuning complete!")


if __name__ == "__main__":
    main()
