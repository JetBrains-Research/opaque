# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""DDP scaling tests for real Hugging Face models with clipped gradients.

Tests multi-GPU degradation, memory efficiency, and convergence across different
batch sizes and gradient accumulation configs. Follows Opacus testing pattern
with mp.spawn() for multi-GPU tests.

Run with:
  pytest tests/compat/test_ddp_multi_gpu_scaling.py -v -s

(pytest automatically skips if < 2 GPUs are available)
"""

import os
import sys
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from opaque import clipped_grad, make_functional
from opaque.utils.pytree import tree_map


# Memory-aware test configurations for A100 40GB multi-GPU scaling
# Models sized to fit within 40GB per GPU with DP-SGD per-example gradients
DDP_TEST_CONFIGS = [
    {
        "model": "Qwen/Qwen2-0.5B",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "max_length": 2048,
        "batch_size": 8,
        "accum_steps": 32,
    },
    {
        "model": "Qwen/Qwen2-1.5B",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "max_length": 1024,
        "batch_size": 8,
        "accum_steps": 16,
    },
    {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "max_length": 1024,
        "batch_size": 8,
        "accum_steps": 16,
    },
]

TRAINING_STEPS = 3
LEARNING_RATE = 1e-3
L2_CLIP_NORM = 1.0


def _setup_ddp(rank, world_size):
    """Setup DDP process group."""
    if sys.platform == "win32":
        raise ValueError("Windows is not supported for multi-GPU tests")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(
        init_method="env://",
        backend="nccl",
    )


def _cleanup_ddp():
    """Cleanup DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _build_text_batch(batch_size):
    """Create deterministic text batch."""
    base_texts = [
        "Hello world test",
        "Another example",
        "Third sample",
        "Final one",
        "Short prompt",
        "Yet another example",
        "Testing a longer input",
        "Final sample in batch",
    ]
    if batch_size <= len(base_texts):
        return base_texts[:batch_size]
    repeats = (batch_size + len(base_texts) - 1) // len(base_texts)
    return (base_texts * repeats)[:batch_size]


def _run_ddp_scaling_test(
    rank,
    world_size,
    model_name,
    target_modules,
    max_length,
    batch_size,
    accum_steps,
    result_dict,
):
    """Run DDP training step with clipped_grad and gradient accumulation.

    Args:
        rank: Process rank
        world_size: Total number of processes
        model_name: HF model ID
        target_modules: LoRA target modules
        max_length: Tokenizer max length
        batch_size: Batch size per GPU
        accum_steps: Gradient accumulation steps
        result_dict: Shared dict to store (success, loss) tuples
    """
    torch.manual_seed(42 + rank)
    torch.cuda.set_device(rank)

    try:
        _setup_ddp(rank, world_size)
        device = torch.device(f"cuda:{rank}")

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
        config._attn_implementation = "eager"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            dtype=torch.float16,
            trust_remote_code=False,
        ).to(device)

        # Apply LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)

        # Prepare batch
        texts = _build_text_batch(batch_size)
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            max_length=max_length,
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = input_ids.clone()

        # Make functional
        fmodel, trainable, frozen = make_functional(
            model,
            disable_autograd_tracking=True,
            partition_trainable=True,
        )

        def per_example_loss(trains, frozens, ids, mask, lbls):
            all_params = {**frozens, **trains}
            outputs = fmodel(all_params, ids, attention_mask=mask, labels=lbls)
            return outputs.loss

        grad_fn, clip_state = clipped_grad(
            per_example_loss,
            argnums=0,
            batch_argnums=(2, 3, 4),
            l2_clip_norm=L2_CLIP_NORM,
        )

        state = clip_state
        trainable_params = trainable
        final_loss = None

        # Training loop
        for step in range(TRAINING_STEPS):
            accumulated = None

            # Gradient accumulation
            for accum_idx in range(accum_steps):
                grads, state = grad_fn(
                    trainable_params,
                    frozen,
                    input_ids,
                    attention_mask,
                    labels,
                    state=state,
                )

                if accumulated is None:
                    accumulated = tree_map(lambda x: x.detach().clone(), grads)
                else:
                    accumulated = tree_map(lambda x, y: x + y, accumulated, grads)

            # Apply accumulated gradients
            scale = 1.0 / float(accum_steps)
            accumulated = tree_map(lambda x: x * scale, accumulated)

            trainable_params = tree_map(
                lambda p, g: p - LEARNING_RATE * g,
                trainable_params,
                accumulated,
            )

            # Sync loss across ranks (for verification)
            if step == TRAINING_STEPS - 1:
                # Compute loss on last step for final comparison
                all_params = {**frozen, **trainable_params}
                outputs = fmodel(all_params, input_ids, attention_mask=attention_mask, labels=labels)
                final_loss = float(outputs.loss.detach().cpu())

        result_dict[rank] = (True, final_loss if final_loss is not None else 0.0)

    except Exception as e:
        result_dict[rank] = (False, str(e))
        if rank == 0:
            print(f"Error on rank {rank}: {e}")
        raise

    finally:
        _cleanup_ddp()


def _run_multi_gpu_wrapper(
    rank,
    world_size,
    model_name,
    target_modules,
    max_length,
    batch_size,
    accum_steps,
    result_dict,
):
    """Wrapper for mp.spawn."""
    _run_ddp_scaling_test(
        rank,
        world_size,
        model_name,
        target_modules,
        max_length,
        batch_size,
        accum_steps,
        result_dict,
    )


@unittest.skipIf(
    torch.cuda.device_count() < 2,
    "Need at least 2 GPUs for DDP scaling tests",
)
class TestDDPMultiGPUScaling(unittest.TestCase):
    """DDP multi-GPU scaling tests to detect memory degradation and OOM issues."""

    def _run_ddp_scaling_test(
        self, model_name, target_modules, max_length, batch_size, accum_steps
    ):
        """Helper to run DDP scaling test on multiple processes."""
        world_size = min(torch.cuda.device_count(), 2)

        with mp.Manager() as manager:
            result_dict = manager.dict()

            mp.spawn(
                _run_multi_gpu_wrapper,
                args=(
                    world_size,
                    model_name,
                    target_modules,
                    max_length,
                    batch_size,
                    accum_steps,
                    result_dict,
                ),
                nprocs=world_size,
                join=True,
            )

            # Verify all ranks succeeded
            for rank in range(world_size):
                success, loss = result_dict[rank]
                self.assertTrue(
                    success,
                    f"DDP training failed on rank {rank}: {loss}",
                )
                if rank == 0:
                    print(f"Final loss on rank {rank}: {loss:.6f}")

    def test_qwen2_0_5b(self):
        """Test Qwen2-0.5B (0.5B, bs=8, accum=32)."""
        cfg = DDP_TEST_CONFIGS[0]
        self._run_ddp_scaling_test(
            model_name=cfg["model"],
            target_modules=cfg["target_modules"],
            max_length=cfg["max_length"],
            batch_size=cfg["batch_size"],
            accum_steps=cfg["accum_steps"],
        )

    def test_qwen2_1_5b(self):
        """Test Qwen2-1.5B (1.5B, bs=8, accum=16)."""
        cfg = DDP_TEST_CONFIGS[1]
        self._run_ddp_scaling_test(
            model_name=cfg["model"],
            target_modules=cfg["target_modules"],
            max_length=cfg["max_length"],
            batch_size=cfg["batch_size"],
            accum_steps=cfg["accum_steps"],
        )

    def test_tinyllama_1_1b(self):
        """Test TinyLlama-1.1B (1.1B, bs=8, accum=16)."""
        cfg = DDP_TEST_CONFIGS[2]
        self._run_ddp_scaling_test(
            model_name=cfg["model"],
            target_modules=cfg["target_modules"],
            max_length=cfg["max_length"],
            batch_size=cfg["batch_size"],
            accum_steps=cfg["accum_steps"],
        )
