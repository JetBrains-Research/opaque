"""Realistic DP-SGD LoRA training with adaptive clipping for Qwen2.5-7B.

Automatically detects and uses available device (CUDA > MPS > CPU).
Uses AG News dataset with realistic settings optimized for 7B model:
- Model: Qwen2.5-7B (7B parameter model)
- Batch size: 2
- Sequence length: 256 tokens
- Training samples: 200
- LoRA rank: 16 with attention layers (q/k/v/o_proj)
- Expected runtime on H200: 5-10 minutes

IMPORTANT - Adaptive Clipping Performance:
- By default: use_adaptive_clipping=False (fast, fixed clip norm)
- Optional: use_adaptive_clipping=True (slow but adaptive)
- Fixed clipping: creates clipped_grad ONCE and reuses it (fast!)
- Adaptive clipping: must recreate clipped_grad each step (slow!)
- This is a known limitation of PyTorch's vmap with changing parameters

Expected timing (on H200 GPU):
DEFAULT (use_adaptive_clipping=False):
- Setup: ~60-90s (includes model loading and one-time clipped_grad creation)
- First step: ~15-25s (first vmap compilation)
- Subsequent steps: ~3-5s each (reuses same function!)
- Total for ~1000 steps (10 epochs × ~100 batches): 5-10 minutes

OPTIONAL (use_adaptive_clipping=True):
- Each step: ~30-60s (recreates clipped_grad + recompilation)
- Not recommended for this configuration
"""

import time

import torch
import torch.nn.functional as F
import torchopt
from datasets import load_dataset
from opaque.optimizers import adaptive_clipping
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM

from opaque.clipping import clipped_grad
from opaque.noise import gaussian_noise
from opaque.utils import make_functional


def compute_causal_lm_loss(logits, targets):
    """Compute causal language modeling loss."""
    # Handle HuggingFace model outputs
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


def main():
    overall_start = time.time()
    print("=" * 80)
    print("DP-SGD LORA + ADAPTIVE CLIPPING - QWEN2.5 + AG NEWS")
    print("=" * 80)

    # Setup device - automatic detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"\n[0/7] Using device: {device} ({device_name})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\n[0/7] Using device: {device} (Apple Silicon)")
        print(f"   MPS built: {torch.backends.mps.is_built()}")
    else:
        device = torch.device("cpu")
        print(f"\n[0/7] Using device: {device}")
        print("   Warning: Training on CPU will be slow")

    use_eager_attention = device.type == "mps"  # Only needed for MPS

    # Setup
    torch.manual_seed(42)
    model_name = "Qwen/Qwen2.5-7B"  # Larger 7B model
    max_seq_len = 256  # Balanced sequence length
    batch_size = 2  # Very small batch size for 7B model memory
    num_train_samples = 200  # Reduced samples due to larger model

    # Load model config and modify dropout settings
    print("\n[1/7] Loading model config...")
    config = AutoConfig.from_pretrained(model_name)

    # Disable dropout for deterministic behavior
    dropout_attrs = [
        "attn_pdrop",
        "resid_pdrop",
        "embd_pdrop",  # GPT-2
        "attention_dropout",
        "hidden_dropout",  # Qwen, LLaMA, Gemma
        "dropout",
        "attn_dropout",
        "ffn_dropout",
    ]
    for attr in dropout_attrs:
        if hasattr(config, attr):
            setattr(config, attr, 0.0)

    # Load model and tokenizer
    print(f"[2/7] Loading model: {model_name}...")
    t0 = time.time()

    # Build model loading kwargs
    model_kwargs = {
        "config": config,
        "trust_remote_code": True,
    }

    # Use eager attention for MPS (no vmap batching rule for SDPA on MPS)
    if use_eager_attention:
        print("   Using eager attention (MPS requires this for vmap compatibility)")
        model_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    print(f"   Model loaded in {time.time() - t0:.1f}s")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Move model to device
    print(f"   Moving model to {device}...")
    t0 = time.time()
    model = model.to(device)
    print(f"   Model moved to {device} in {time.time() - t0:.1f}s")

    # Apply LoRA
    print("[3/7] Applying LoRA...")
    t0 = time.time()
    lora_config = LoraConfig(
        r=16,  # Moderate LoRA rank for memory efficiency
        lora_alpha=32,  # LoRA scaling factor
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],  # Apply LoRA to attention layers only
        lora_dropout=0.0,  # No dropout for deterministic behavior
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(f"   LoRA applied in {time.time() - t0:.1f}s")

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load real dataset - AG News (news articles)
    print("[4/7] Loading dataset...")
    print("   Downloading/loading AG News dataset (may take 30s-2min first time)...")
    t0 = time.time()
    dataset = load_dataset("ag_news", split="train")
    print(f"   Dataset loaded in {time.time() - t0:.1f}s")
    print("   Dataset: ag_news (news articles)")
    print(f"   Total examples in dataset: {len(dataset)}")

    # Extract text from AG News
    print(f"   Selecting {num_train_samples} samples...")
    t0 = time.time()
    all_texts = [f"{item['text']}" for item in dataset.select(range(num_train_samples))]
    print(f"   Selection completed in {time.time() - t0:.1f}s")

    print(f"   Total news articles: {len(all_texts)}")
    print(f"   Batch size: {batch_size}")
    print(f"   Number of batches: {len(all_texts) // batch_size}")
    print(f"   Sample text preview: {all_texts[0][:100]}...")

    # Tokenize all samples upfront
    print("   Tokenizing all texts...")
    t0 = time.time()
    all_encodings = tokenizer(
        all_texts,
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    all_tokens = all_encodings["input_ids"].to(device)
    print(f"   All tokens shape: {all_tokens.shape}")
    print(f"   Tokenization completed in {time.time() - t0:.1f}s")

    # Create batches
    print("   Creating batches...")
    t0 = time.time()
    num_batches = len(all_texts) // batch_size
    batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_tokens = all_tokens[start_idx:end_idx]
        batches.append(batch_tokens)

    print(
        f"   Created {len(batches)} batches of size {batch_size} in {time.time() - t0:.1f}s"
    )

    # Convert to functional (only LoRA parameters)
    print("[5/7] Converting to functional form (LoRA parameters only)...")
    t0 = time.time()

    # Freeze base model and only train LoRA parameters
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    fmodel, params = make_functional(model, disable_autograd_tracking=True)
    param_names = [
        name for name, param in model.named_parameters() if param.requires_grad
    ]
    print(f"   Number of trainable LoRA parameters: {len(param_names)}")
    print(f"   Total parameter tensors: {len(params)}")
    print(f"   Conversion completed in {time.time() - t0:.1f}s")

    # Define per-example loss
    def per_example_loss_fn(params_tuple, tokens_single):
        # Add batch dimension
        tokens_batch = tokens_single.unsqueeze(0)
        logits = fmodel(params_tuple, tokens_batch)
        return compute_causal_lm_loss(logits, tokens_batch)

    # Setup DP-SGD training with adaptive clipping
    print("[6/7] Setting up DP-SGD with adaptive clipping...")
    initial_clip_norm = 1.0
    learning_rate = 0.00001  # Learning rate for smaller batch size
    num_epochs = 10  # More epochs due to very small batch size
    noise_multiplier = 0.24
    target_clip_rate = 0.20
    use_adaptive_clipping = (
        False  # Set to True for adaptive clipping (slow: ~1min/step)
    )

    print(f"   Clip norm: {initial_clip_norm}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Noise multiplier: {noise_multiplier}")
    print(f"   Adaptive clipping: {use_adaptive_clipping}")
    if use_adaptive_clipping:
        print(f"   Target clip rate: {target_clip_rate:.1%}")
        print("   ⚠️  WARNING: Adaptive clipping causes ~1min recompilation per step!")
        print("   ⚠️  Set use_adaptive_clipping=False for faster training.")
    else:
        print("   ✓ Using fixed clipping (fast mode)")
    print(f"   Number of epochs: {num_epochs}")
    print(f"   Batches per epoch: {len(batches)}")
    print(f"   Total training steps: {num_epochs * len(batches)}")

    # Create base optimizer and optionally wrap with adaptive clipping
    base_opt = torchopt.sgd(lr=learning_rate)

    if use_adaptive_clipping:
        init_fn, step_fn = adaptive_clipping(
            base_opt,
            initial_clip_norm=initial_clip_norm,
            target_clip_rate=target_clip_rate,
            clip_norm_max=10000,
        )
        opt_state = init_fn(params)
        fixed_clip_norm = None  # Will use adaptive norm from state
    else:
        # Use base optimizer directly (no adaptive clipping wrapper)
        # This avoids recompilation since clip_norm stays constant
        opt_state = base_opt.init(params)
        fixed_clip_norm = initial_clip_norm  # Use constant clip norm
        print(f"   Using fixed clip norm (no adaptation): {fixed_clip_norm}")

    # Note: Noise generation uses stateful API with per-step seed for reproducibility

    # Pre-create clipped_grad function if using fixed clipping (avoids recreation)
    if not use_adaptive_clipping:
        print(
            "\n   Creating clipped_grad function (one-time setup for fixed clipping)..."
        )
        t0 = time.time()
        fixed_clipped_grad_fn = clipped_grad(
            per_example_loss_fn,
            argnums=0,
            batch_argnums=(1,),
            l2_clip_norm=fixed_clip_norm,
            microbatch_size=1,  # Microbatch size of 1 for 7B model
            keep_batch_dim=False,
            return_grad_norms=True,
            return_values=True,
        )
        print(f"   ✓ Created in {time.time() - t0:.1f}s")
    else:
        fixed_clipped_grad_fn = None

    # Training loop
    setup_time = time.time() - overall_start
    print("\n[7/7] Running DP-SGD training loop...")
    print("=" * 80)
    print(f"\n✓ Setup completed in {setup_time:.1f}s")
    print("\nNote: First step will compile the model (this is normal)")
    print("Subsequent steps should be fast (~5-10s each)\n")

    try:
        losses = []
        clip_norms_history = []
        clip_rates_history = []
        global_step = 0

        for epoch in range(num_epochs):
            print(f"\n{'=' * 80}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'=' * 80}")
            epoch_losses = []

            for batch_idx, tokens in enumerate(batches):
                if global_step == 0:
                    print("Step 1: Compiling model...")
                    print("  - Setting up vmap for per-example gradients")
                    print("  - Please wait...\n")

                # Determine clip norm to use
                current_clip_norm = (
                    fixed_clip_norm
                    if fixed_clip_norm is not None
                    else opt_state.current_clip_norm
                )

                # 1. Compute clipped gradients
                if global_step <= 2:
                    print(f"\n  Step {global_step + 1} - Detailed timing:")

                if fixed_clipped_grad_fn is not None:
                    # Use pre-created function (fixed clipping - no recreation!)
                    grad_compute_start = time.time()
                    grads_tuple, aux = fixed_clipped_grad_fn(params, tokens)
                    grad_compute_time = time.time() - grad_compute_start
                    if global_step <= 2:
                        print(
                            f"    gradient computation (reusing fn): {grad_compute_time:.1f}s"
                        )
                else:
                    # Adaptive clipping - must recreate with new clip_norm
                    fn_create_start = time.time()
                    clipped_grad_fn = clipped_grad(
                        per_example_loss_fn,
                        argnums=0,
                        batch_argnums=(1,),
                        l2_clip_norm=current_clip_norm,
                        microbatch_size=1,  # Microbatch size of 1 for 7B model
                        keep_batch_dim=False,
                        return_grad_norms=True,
                        return_values=True,
                    )
                    fn_create_time = time.time() - fn_create_start
                    if global_step <= 2:
                        print(f"    clipped_grad creation: {fn_create_time:.1f}s")

                    grad_compute_start = time.time()
                    grads_tuple, aux = clipped_grad_fn(params, tokens)
                    grad_compute_time = time.time() - grad_compute_start
                    if global_step <= 2:
                        print(f"    gradient computation: {grad_compute_time:.1f}s")

                # 2. Add Gaussian noise for DP (stateful API)
                stddev = noise_multiplier * current_clip_norm
                noise_fn, noise_state = gaussian_noise(stddev=stddev, generator=42 + global_step)
                noisy_grads, noise_state = noise_fn(grads_tuple, noise_state)

                # 3. Optimizer step
                if use_adaptive_clipping:
                    # Use adaptive clipping wrapper
                    updates, opt_state, metrics = step_fn(
                        noisy_grads, aux.grad_norms, opt_state, params=params
                    )
                else:
                    # Use base optimizer directly (fixed clipping)
                    updates, opt_state = base_opt.update(
                        noisy_grads, opt_state, params=params
                    )
                    # Manually compute metrics
                    clip_rate = (
                        (aux.grad_norms > current_clip_norm).float().mean().item()
                    )
                    metrics = {
                        "clip_norm": current_clip_norm,
                        "clip_rate": clip_rate,
                        "step": global_step + 1,
                    }

                # 4. Apply updates to get new parameters
                params = torchopt.apply_updates(params, updates)

                # Track metrics
                avg_loss = aux.values.mean().item()
                losses.append(avg_loss)
                epoch_losses.append(avg_loss)
                clip_norms_history.append(metrics["clip_norm"])
                clip_rates_history.append(metrics["clip_rate"])

                global_step += 1

                # Print step metrics
                print(
                    f"Step {global_step:3d} [Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(batches)}]: "
                    f"loss={avg_loss:.4f}, "
                    f"clip_norm={metrics['clip_norm']:.3f}, "
                    f"clip_rate={metrics['clip_rate']:.1%}, "
                    f"grad_norm(min={aux.grad_norms.min().item():.2f}, "
                    f"max={aux.grad_norms.max().item():.2f})"
                )

            # Print epoch summary
            epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(
                f"\n→ Epoch {epoch + 1} summary: avg_loss={epoch_avg_loss:.4f}, "
                f"clip_norm={metrics['clip_norm']:.3f}, "
                f"clip_rate={metrics['clip_rate']:.1%}"
            )

        # Success!
        print("\n" + "=" * 80)
        print(
            f"✅ DP-SGD LORA TRAINING COMPLETE! (ADAPTIVE CLIPPING + {device.type.upper()})"
        )
        print("=" * 80)
        print(f"Device: {device}")
        print(f"Dataset: ag_news ({len(all_texts)} news articles)")
        print(f"Batch size: {batch_size}, Number of batches: {len(batches)}")
        print(f"Sequence length: {all_tokens.shape[1]}")
        print("\nLoRA configuration:")
        print(f"  Rank (r): {lora_config.r}")
        print(f"  Alpha: {lora_config.lora_alpha}")
        print(f"  Target modules: {lora_config.target_modules}")
        print(f"  Trainable LoRA parameters: {len(param_names)}")
        print("\nTraining results:")
        print(f"  Total epochs: {num_epochs}")
        print(f"  Total steps: {global_step}")
        print(f"  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss: {losses[-1]:.4f}")
        print(
            f"  Loss reduction: {losses[0] - losses[-1]:.4f} ({(1 - losses[-1] / losses[0]) * 100:.1f}%)"
        )
        if use_adaptive_clipping:
            print("\nAdaptive clipping:")
            print(f"  Initial clip norm: {initial_clip_norm:.3f}")
            print(f"  Final clip norm: {clip_norms_history[-1]:.3f}")
            print(f"  Target clip rate: {target_clip_rate:.1%}")
            print(f"  Final clip rate: {clip_rates_history[-1]:.1%}")
            print(
                f"  Clip norm range: [{min(clip_norms_history):.3f}, {max(clip_norms_history):.3f}]"
            )
        else:
            print("\nFixed clipping:")
            print(f"  Clip norm: {fixed_clip_norm:.3f} (constant)")
            print(
                f"  Average clip rate: {sum(clip_rates_history) / len(clip_rates_history):.1%}"
            )
        print("\nDP parameters:")
        print(f"  Noise multiplier: {noise_multiplier}")
        final_clip = (
            clip_norms_history[-1] if use_adaptive_clipping else fixed_clip_norm
        )
        print(f"  Final noise stddev: {noise_multiplier * final_clip:.4f}")

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ TRAINING FAILED")
        print("=" * 80)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)[:200]}")
        import traceback

        print("\nFull traceback:")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
