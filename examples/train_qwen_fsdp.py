"""DP-SGD LoRA training with FSDP for Qwen2-0.5B across multiple GPUs.

This script demonstrates:
1. FSDP for sharding 0.5B model across GPUs (scales to 8B+ with proper setup)
2. DP-SGD with per-example gradient clipping
3. Distributed noise generation (deterministic, seed-based)
4. LoRA parameter-efficient fine-tuning
5. AG News dataset for realistic training

Launch with torch.distributed.run:
    python -m torch.distributed.run --nproc_per_node=4 examples/train_qwen_fsdp.py

Or for single GPU (no FSDP):
    python examples/train_qwen_fsdp.py

Expected performance on 4x L4 GPUs:
- Setup: ~30-60s (model loading, FSDP wrapping, tokenization)
- First step: ~10-20s (vmap compilation)
- Subsequent steps: ~2-3s each
- Total training (3 epochs): ~5-10 minutes
"""

import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchopt
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from opaque.clipping import clipped_grad
from opaque.distributed import (
    get_rank,
    get_world_size,
    is_initialized,
    wrap_model_for_dp_fsdp,
)
from opaque.noise import gaussian_noise
from opaque.utils import make_functional, merge


def compute_causal_lm_loss(logits, targets):
    """Compute causal language modeling loss."""
    if hasattr(logits, "logits"):
        logits = logits.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, logits.size(-1)), shift_targets.view(-1))
    return loss


def setup_distributed():
    """Initialize distributed training if available."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # Initialize process group
        dist.init_process_group(backend="nccl")
        
        # Set device for this process
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        
        return True, rank, world_size, device
    else:
        # Single GPU mode
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return False, 0, 1, device


def main():
    overall_start = time.time()
    
    # Setup distributed training
    distributed, rank, world_size, device = setup_distributed()
    use_fsdp = distributed and os.environ.get("OPAQUE_USE_FSDP", "1") == "1"
    
    is_main = rank == 0  # Only rank 0 prints
    
    if is_main:
        print("=" * 80)
        print("DP-SGD LORA + FSDP - QWEN2-0.5B + AG NEWS")
        print("=" * 80)
        print(f"\nDistributed Setup:")
        mode = "FSDP (Multi-GPU)" if use_fsdp else ("DDP-style (Multi-GPU)" if distributed else "Single GPU")
        print(f"  Mode: {mode}")
        print(f"  World size: {world_size}")
        print(f"  Rank: {rank}")
        print(f"  Device: {device}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(device)}")
            print(f"  Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    # Configuration
    torch.manual_seed(42 + rank)  # Different seed per rank for data diversity
    model_name = "Qwen/Qwen2-0.5B"  # Smaller model that fits in memory before sharding
    max_seq_len = 128
    batch_size = 1
    num_train_samples = 400  # More samples
    learning_rate = 0.0001
    num_epochs = 3  # Fewer epochs for quick test
    noise_multiplier = 0.24
    initial_clip_norm = 1.0
    
    # Load model config
    if is_main:
        print("\n[1/7] Loading model config...")
    
    config = AutoConfig.from_pretrained(model_name)
    
    # Disable dropout
    dropout_attrs = ["attention_dropout", "hidden_dropout", "dropout"]
    for attr in dropout_attrs:
        if hasattr(config, attr):
            setattr(config, attr, 0.0)
    
    # Load model
    if is_main:
        print(f"[2/7] Loading model: {model_name}...")
        t0 = time.time()
    
    # Load on CPU first (FSDP will handle moving to GPUs)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        attn_implementation="eager",  # Required for vmap
        torch_dtype=torch.float32,  # Load in FP32, FSDP will convert if mixed_precision=True
    )
    
    if is_main:
        print(f"   Model loaded in {time.time() - t0:.1f}s")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # DON'T move to device yet - FSDP will handle it
    
    # Apply LoRA
    if is_main:
        print("[3/7] Applying LoRA...")
        t0 = time.time()
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    if is_main:
        model.print_trainable_parameters()
        print(f"   LoRA applied in {time.time() - t0:.1f}s")
    
    # Freeze base model parameters for LoRA training (saves memory in DP)
    train_full_model = False
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
    
    # Wrap with FSDP if enabled, else move to device directly
    if use_fsdp:
        if is_main:
            print("\n[FSDP] Wrapping model with Fully Sharded Data Parallel...")
            t0 = time.time()

        # FSDP will respect requires_grad settings (already set by PEFT)
        model = wrap_model_for_dp_fsdp(
            model,
            sharding_strategy="FULL_SHARD",  # Maximum memory savings
            mixed_precision=True,  # Use mixed precision for efficiency
            auto_wrap_min_params=int(1e9),  # Avoid per-layer wrapping for stability
            use_orig_params=not train_full_model,  # Full fine-tuning needs uniform grads
        )

        if is_main:
            print(f"   FSDP wrapping completed in {time.time() - t0:.1f}s")
            print(f"   Sharding strategy: FULL_SHARD (parameters + gradients + optimizer states)")
            print(f"   Mixed precision: Enabled (reduces memory further)")
            print(f"   Model sharded across {world_size} GPUs")
    else:
        # DDP-style or single GPU: move model to device directly
        if is_main:
            print(f"\n   Moving model to {device}...")
            t0 = time.time()
        model = model.to(device)
        if is_main:
            print(f"   Model moved in {time.time() - t0:.1f}s")
    
    # Load dataset
    if is_main:
        print("\n[4/7] Loading dataset...")
        print("   Downloading/loading AG News dataset...")
        t0 = time.time()
    
    dataset = load_dataset("ag_news", split="train")
    
    if is_main:
        print(f"   Dataset loaded in {time.time() - t0:.1f}s")
        print(f"   Total examples: {len(dataset)}")
    
    # Extract and tokenize samples
    if is_main:
        print(f"   Selecting {num_train_samples} samples...")
    
    # Each rank gets different samples for data parallelism
    samples_per_rank = num_train_samples // world_size
    start_idx = rank * samples_per_rank
    end_idx = start_idx + samples_per_rank
    
    all_texts = [item["text"] for item in dataset.select(range(start_idx, end_idx))]
    
    if is_main:
        print(f"   Samples per rank: {samples_per_rank}")
        print(f"   Rank {rank} range: [{start_idx}, {end_idx})")
    
    # Tokenize
    if is_main:
        print("   Tokenizing texts...")
        t0 = time.time()
    
    all_encodings = tokenizer(
        all_texts,
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    all_tokens = all_encodings["input_ids"].to(device)
    
    if is_main:
        print(f"   Tokens shape (per rank): {all_tokens.shape}")
        print(f"   Tokenization completed in {time.time() - t0:.1f}s")
    
    # Create batches
    num_batches = len(all_texts) // batch_size
    batches = []
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batches.append(all_tokens[start:end])
    
    if is_main:
        print(f"   Created {len(batches)} batches of size {batch_size}")
    
    # Convert to functional
    if is_main:
        print("\n[5/7] Converting to functional form...")
        t0 = time.time()
    
    fmodel, trainable_params, frozen_params = make_functional(
        model,
        disable_autograd_tracking=False,
        partition_trainable=True,
    )
    
    if is_main:
        print(f"   Conversion completed in {time.time() - t0:.1f}s")
        print(f"   Trainable parameters: {len(trainable_params)}")
    
    # Define per-example loss
    def per_example_loss_fn(params_dict, tokens_single):
        tokens_batch = tokens_single.unsqueeze(0)
        all_params = merge(frozen_params, params_dict)
        logits = fmodel(all_params, tokens_batch)
        return compute_causal_lm_loss(logits, tokens_batch)
    
    # Setup DP-SGD
    if is_main:
        print("\n[6/7] Setting up DP-SGD...")
        print(f"   Clip norm: {initial_clip_norm}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Noise multiplier: {noise_multiplier}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Batches per epoch: {len(batches)}")
        print(f"   Total steps: {num_epochs * len(batches)}")
    
    # Create optimizer
    base_opt = torchopt.sgd(lr=learning_rate)
    opt_state = base_opt.init(trainable_params)
    
    # Create clipped_grad function (fixed clipping for speed)
    if is_main:
        print("\n   Creating clipped_grad function...")
        t0 = time.time()
    
    clipped_grad_fn, clip_state = clipped_grad(
        per_example_loss_fn,
        argnums=0,
        batch_argnums=(1,),
        l2_clip_norm=initial_clip_norm,
        microbatch_size=1,
        keep_batch_dim=False,
        return_grad_norms=True,
        return_values=True,
    )
    
    if is_main:
        print(f"   Created in {time.time() - t0:.1f}s")
    
    # Create noise function (different seed per rank for independent noise)
    if is_main:
        print("\n   Creating noise function...")
    
    stddev = noise_multiplier * initial_clip_norm
    noise_gen = torch.Generator().manual_seed(42 + rank)
    noise_fn, noise_state = gaussian_noise(stddev=stddev, generator=noise_gen)
    
    # Training loop
    if is_main:
        setup_time = time.time() - overall_start
        print("\n[7/7] Running DP-SGD training loop...")
        print("=" * 80)
        print(f"\n✓ Setup completed in {setup_time:.1f}s")
        print("\nStarting training...\n")
    
    if distributed:
        dist.barrier()  # Sync all ranks before training
    
    try:
        losses = []
        global_step = 0
        
        for epoch in range(num_epochs):
            if is_main:
                print(f"\n{'='*80}")
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"{'='*80}")
            
            epoch_losses = []
            
            for batch_idx, tokens in enumerate(batches):
                step_start = time.time()
                
                if global_step == 0 and is_main:
                    print(f"Step 1: Compiling model (first step)...")
                    print(f"  This will take 30-60s, subsequent steps ~5-10s each\n")
                
                # 1. Compute clipped gradients
                (grads_tuple, aux), clip_state = clipped_grad_fn(
                    trainable_params,
                    tokens,
                    state=clip_state,
                )
                
                # 2. Add noise (different noise per rank)
                noisy_grads, noise_state = noise_fn(grads_tuple, noise_state)
                
                # 3. If distributed, average gradients across ranks
                if distributed:
                    # FSDP handles gradient synchronization automatically
                    # But for functional API, we need to average manually
                    from opaque.distributed import average_gradients
                    noisy_grads = average_gradients(noisy_grads)
                
                # 4. Optimizer step
                updates, opt_state = base_opt.update(
                    noisy_grads,
                    opt_state,
                    params=trainable_params,
                )
                trainable_params = torchopt.apply_updates(trainable_params, updates)
                
                # Track metrics
                avg_loss = aux.loss_values.mean().item()
                losses.append(avg_loss)
                epoch_losses.append(avg_loss)
                
                clip_rate = (aux.grad_norms > initial_clip_norm).float().mean().item()
                
                global_step += 1
                step_time = time.time() - step_start
                
                # Print every 10 steps
                if is_main and (global_step % 10 == 0 or global_step == 1):
                    print(
                        f"Step {global_step:3d} [Epoch {epoch+1}, Batch {batch_idx+1}/{len(batches)}]: "
                        f"loss={avg_loss:.4f}, "
                        f"clip_rate={clip_rate:.1%}, "
                        f"time={step_time:.1f}s"
                    )
            
            # Epoch summary
            if is_main:
                epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
                print(
                    f"\n→ Epoch {epoch + 1} summary: "
                    f"avg_loss={epoch_avg_loss:.4f}"
                )
        
        # Final summary
        if is_main:
            print("\n" + "=" * 80)
            print(f"✅ DP-SGD LORA + FSDP TRAINING COMPLETE!")
            print("=" * 80)
            print(f"\nDistributed configuration:")
            print(f"  World size: {world_size} GPUs")
            print(f"  Sharding: FSDP FULL_SHARD")
            print(f"  Samples per rank: {samples_per_rank}")
            print(f"  Total samples: {num_train_samples}")
            print(f"\nTraining results:")
            print(f"  Total epochs: {num_epochs}")
            print(f"  Total steps: {global_step}")
            print(f"  Initial loss: {losses[0]:.4f}")
            print(f"  Final loss: {losses[-1]:.4f}")
            print(f"  Loss reduction: {losses[0] - losses[-1]:.4f} ({(1 - losses[-1]/losses[0]) * 100:.1f}%)")
            print(f"\nDP parameters:")
            print(f"  Clip norm: {initial_clip_norm}")
            print(f"  Noise multiplier: {noise_multiplier}")
            print(f"  Noise stddev: {noise_multiplier * initial_clip_norm:.4f}")
            
            total_time = time.time() - overall_start
            print(f"\nTotal time: {total_time/60:.1f} minutes")
            print(f"Average time per step: {total_time/global_step:.1f}s")
    
    except Exception as e:
        if is_main:
            print("\n" + "=" * 80)
            print("❌ TRAINING FAILED")
            print("=" * 80)
            print(f"Error: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        if distributed:
            dist.destroy_process_group()
    
    return 0


if __name__ == "__main__":
    exit(main())
