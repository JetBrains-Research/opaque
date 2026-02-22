"""Clean DDP-style DP-SGD LoRA test (no FSDP).

Purpose:
- Minimal distributed test that proves DP-SGD training runs end-to-end.
- Uses DDP-style data parallelism with manual gradient averaging.

Launch:
  uv run python -m torch.distributed.run --nproc_per_node=4 examples/train_qwen_ddp.py

Single GPU:
  uv run python examples/train_qwen_ddp.py
"""

import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchopt
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from opaque.clipping import adaptive_clipped_grad
from opaque.distributed import sum_gradients
from opaque.noise import gaussian_noise
from opaque.random import key
from opaque.utils import make_functional, merge


def compute_causal_lm_loss(logits, targets):
    """Compute causal language modeling loss."""
    if hasattr(logits, "logits"):
        logits = logits.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, logits.size(-1)), shift_targets.view(-1)
    )


def setup_distributed():
    """Initialize distributed training if launched via torch.distributed.run."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return True, rank, world_size, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, 1, device


def main():
    start = time.time()
    distributed, rank, world_size, device = setup_distributed()
    is_main = rank == 0

    if is_main:
        print("=" * 80)
        print("DDP-STYLE DP-SGD LORA TEST - QWEN2-0.5B + AG NEWS")
        print("=" * 80)
        print(f"Distributed: {distributed}, World size: {world_size}, Device: {device}")

    torch.manual_seed(42 + rank)

    model_name = "Qwen/Qwen2-0.5B"
    max_seq_len = 2048
    logical_batch_size = 4
    microbatch_size = 2
    num_train_samples = 128
    num_epochs = 2
    learning_rate = 1e-4
    noise_multiplier = 0.24
    initial_clip_norm = 1.0
    target_clip_rate = 0.75
    clip_learning_rate = 0.2

    if is_main:
        print("\n[1/6] Loading config + model...")
    config = AutoConfig.from_pretrained(model_name)
    for attr in ["attention_dropout", "hidden_dropout", "dropout"]:
        if hasattr(config, attr):
            setattr(config, attr, 0.0)

    model_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=model_dtype,
    ).to(device)

    # Reduce activation memory for large logical batches
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main:
        print("[2/6] Applying LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    # Disable input requires_grad hooks that break functorch transforms.
    if hasattr(model, "disable_input_require_grads"):
        model.disable_input_require_grads()
    if hasattr(model, "_require_grads_hook") and model._require_grads_hook is not None:
        model._require_grads_hook.remove()
        model._require_grads_hook = None
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    if is_main:
        model.print_trainable_parameters()

    if is_main:
        print("[3/6] Loading dataset...")
    dataset = load_dataset("ag_news", split="train")
    samples_per_rank = num_train_samples // world_size
    start_idx = rank * samples_per_rank
    end_idx = start_idx + samples_per_rank
    texts = [item["text"] for item in dataset.select(range(start_idx, end_idx))]

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    tokens = enc["input_ids"].to(device)

    batches = [
        tokens[i : i + logical_batch_size]
        for i in range(0, len(tokens), logical_batch_size)
    ]

    if is_main:
        print("[4/6] Converting to functional form...")
    fmodel, trainable, frozen = make_functional(
        model,
        disable_autograd_tracking=False,
        partition_trainable=True,
    )

    def per_example_loss_fn(trainable_params, tokens_single):
        tokens_batch = tokens_single.unsqueeze(0)
        all_params = merge(frozen, trainable_params)
        logits = fmodel(all_params, tokens_batch)
        return compute_causal_lm_loss(logits, tokens_batch)

    if is_main:
        print("[5/6] Setting up DP-SGD...")
    grad_fn, clip_state = adaptive_clipped_grad(
        per_example_loss_fn,
        argnums=0,
        batch_argnums=(1,),
        initial_clip_norm=initial_clip_norm,
        target_quantile=target_clip_rate,
        learning_rate=clip_learning_rate,
        microbatch_size=microbatch_size,
        keep_batch_dim=False,
        return_aux=True,
        key=key(42),
    )
    # Same seed on all devices for sharded Poisson sampling
    # When distributed is detected, noise automatically uses the same seed everywhere
    # (no need to manually shift by rank)
    noise_fn, noise_state = gaussian_noise(
        stddev=noise_multiplier * clip_state.sensitivity(),
        key=key(42),
    )

    # TorchOpt functional optimizer (same initialization on all devices)
    opt = torchopt.sgd(lr=learning_rate)
    opt_state = opt.init(trainable)
    # Optimizer state stays synchronized automatically:
    #    - opt.update() is a pure function (same inputs -> same outputs)
    #    - After sum_gradients() + noise (same seed), all devices have identical noisy_grads
    #    - Therefore, opt_state evolves identically on all devices (no explicit synchronization needed)

    if distributed:
        dist.barrier()

    if is_main:
        print("[6/6] Training...\n")

    global_step = 0
    for epoch in range(num_epochs):
        if is_main:
            print(f"Epoch {epoch + 1}/{num_epochs}")
        for batch in batches:
            (grads, aux), clip_state = grad_fn(trainable, batch, state=clip_state)

            # Standard DP-SGD with sharded Poisson sampling:
            # 1. Aggregate clipped gradients across devices
            if distributed:
                grads = sum_gradients(grads)  # Sum before noise

            # 2. Add noise on EVERY device (all with same seed -> same noise)
            # IMPORTANT: noise_fn() is called on EVERY device in the distributed setting
            #    NOT just the main rank. Each device independently applies the same noise.
            noisy_grads, noise_state = noise_fn(grads, noise_state)

            # 3. Update parameters with noisy aggregated gradients
            updates, opt_state = opt.update(noisy_grads, opt_state, params=trainable)
            trainable = torchopt.apply_updates(trainable, updates)
            # opt_state is now identical on all devices (pure function property)
            # No explicit synchronization needed

            if is_main and global_step % 20 == 0:
                loss_val = aux.loss_values.mean().item()
                clip_rate = (
                    (aux.grad_norms > clip_state.clip_norm).float().mean().item()
                )
                print(
                    f"Step {global_step:4d}: loss={loss_val:.4f}, "
                    f"clip_rate={clip_rate:.1%}, clip_norm={clip_state.clip_norm:.3f}"
                )

            global_step += 1

    if is_main:
        elapsed = time.time() - start
        print("\nDDP test complete")
        print(f"Steps: {global_step}, Time: {elapsed:.1f}s")

    if distributed:
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
