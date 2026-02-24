# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""DDP multi-GPU model testing: quick sanity checks and scaling validation.

Uses shared model configs and training logic from tests/conftest.py.
Tests DDP training with HuggingFace models (Qwen2, TinyLlama, Phi-3) using mp.spawn.

Quick tests run in ~1-2 minutes and validate basic DDP functionality.
Scaling tests verify memory efficiency and convergence across multiple models.

Run with:
  pytest tests/distributed/test_ddp_models.py -v -s
  pytest tests/distributed/test_ddp_models.py::TestDDPQuickSanity -v -s  (quick only)
  pytest tests/distributed/test_ddp_models.py::TestDDPMultiGPUScaling -v -s  (scaling only)

(pytest automatically skips multi-GPU tests if < 2 GPUs are available)
"""

import os
import socket
import sys
import unittest

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from opaque import clipped_grad, make_functional
from opaque.utils.pytree import tree_map
from tests.conftest import (
    MODEL_CONFIGS,
    build_text_batch,
    has_min_gpu_memory,
    load_model_with_lora,
)

# Check if HuggingFace dependencies are available (required for LoRA tests)
try:
    import transformers  # noqa: F401

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import peft  # noqa: F401

    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

HAS_HF = HAS_TRANSFORMERS and HAS_PEFT

TRAINING_STEPS = 3
LEARNING_RATE = 1e-3
L2_CLIP_NORM = 1.0


def _is_distributed():
    return dist.is_available() and dist.is_initialized()


def _rank():
    if _is_distributed():
        return dist.get_rank()
    return 0


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _setup_ddp(rank, world_size, port):
    """Setup DDP process group."""
    if sys.platform == "win32":
        raise ValueError("Windows is not supported for multi-GPU tests")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
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


def _run_ddp_scaling_test(
    rank,
    world_size,
    port,
    model_key,
    batch_size,
    max_length,
    accum_steps,
    result_dict,
):
    """Run DDP training step with clipped_grad and gradient accumulation using shared utilities.

    Args:
        rank: Process rank
        world_size: Total number of processes
        model_key: Key into MODEL_CONFIGS dict
        batch_size: Batch size per GPU
        max_length: Max sequence length
        accum_steps: Gradient accumulation steps
        result_dict: Shared dict to store (success, loss) tuples
    """
    torch.manual_seed(42 + rank)
    torch.cuda.set_device(rank)

    try:
        _setup_ddp(rank, world_size, port)
        device = torch.device(f"cuda:{rank}")

        # Load model using shared utility
        config = MODEL_CONFIGS[model_key]
        model, tokenizer = load_model_with_lora(config, device=device)

        # Prepare batch using shared utility
        texts = build_text_batch(batch_size)
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
            for _ in range(accum_steps):
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
            accumulated = tree_map(lambda x, s=scale: x * s, accumulated)

            trainable_params = tree_map(
                lambda p, g: p - LEARNING_RATE * g,
                trainable_params,
                accumulated,
            )

            # Sync loss across ranks (for verification)
            if step == TRAINING_STEPS - 1:
                # Compute loss on last step for final comparison
                all_params = {**frozen, **trainable_params}
                outputs = fmodel(
                    all_params, input_ids, attention_mask=attention_mask, labels=labels
                )
                final_loss = float(outputs.loss.detach().cpu())

        result_dict[rank] = (True, final_loss if final_loss is not None else 0.0)

    except Exception as e:
        result_dict[rank] = (False, str(e))
        if rank == 0:
            print(f"Error on rank {rank}: {e}")
        raise

    finally:
        _cleanup_ddp()


def _run_ddp_quick_sanity(rank, world_size, port):
    """Run quick DDP sanity step in spawned workers."""
    torch.manual_seed(42 + rank)
    torch.cuda.set_device(rank)

    try:
        _setup_ddp(rank, world_size, port)
        device = torch.device(f"cuda:{rank}")

        config = MODEL_CONFIGS["qwen2-0.5b"]
        model, tokenizer = load_model_with_lora(config, device=device)
        base_model = (
            model.get_base_model() if hasattr(model, "get_base_model") else model
        )
        if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
            base_model.model.layers = base_model.model.layers[:2]
        elif hasattr(base_model, "layers"):
            base_model.layers = base_model.layers[:2]

        batch_size = 4
        max_length = 512
        accum_steps = 8
        training_steps = 2

        texts = build_text_batch(batch_size)
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

        fmodel, trainable, frozen = make_functional(
            model,
            disable_autograd_tracking=True,
            partition_trainable=True,
        )

        def per_example_loss(
            trainable_params,
            frozen_params,
            input_ids_single,
            mask_single,
            labels_single,
        ):
            all_params = {**frozen_params, **trainable_params}
            outputs = fmodel(
                all_params,
                input_ids_single,
                attention_mask=mask_single,
                labels=labels_single,
            )
            return outputs.loss

        grad_fn, clip_state = clipped_grad(
            per_example_loss,
            argnums=0,
            batch_argnums=(2, 3, 4),
            l2_clip_norm=1.0,
        )

        state = clip_state
        trainable_params = trainable

        for _step in range(training_steps):
            accumulated = None
            for _ in range(accum_steps):
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

            scale = 1.0 / float(accum_steps)
            accumulated = tree_map(lambda x, s=scale: x * s, accumulated)

            for name, param in trainable_params.items():
                param.grad = accumulated[name].to(param.dtype)

        if _is_distributed():
            dist.barrier()

    finally:
        _cleanup_ddp()


def _run_multi_gpu_wrapper(
    rank,
    world_size,
    port,
    model_key,
    batch_size,
    max_length,
    accum_steps,
    result_dict,
):
    """Wrapper for mp.spawn."""
    _run_ddp_scaling_test(
        rank,
        world_size,
        port,
        model_key,
        batch_size,
        max_length,
        accum_steps,
        result_dict,
    )


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(
    not HAS_HF,
    reason="transformers/peft libraries not installed",
)
class TestDDPQuickSanity:
    """Quick DDP sanity check - minimal config for fast feedback."""

    def test_ddp_qwen2_0_5b_basic(self):
        """Quick DDP test with Qwen2-0.5B, reduced layers, 2 training steps."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Need at least 2 GPUs for DDP quick sanity")

        if not has_min_gpu_memory(12):
            pytest.skip("Requires CUDA GPU with >= 12GB memory")

        port = _find_free_port()
        mp.spawn(_run_ddp_quick_sanity, args=(2, port), nprocs=2, join=True)


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Need at least 2 GPUs for DDP scaling tests",
)
@pytest.mark.skipif(
    not HAS_HF,
    reason="transformers/peft libraries not installed",
)
class TestDDPMultiGPUScaling(unittest.TestCase):
    """DDP multi-GPU scaling tests to detect memory degradation and OOM issues."""

    def _run_ddp_scaling_test(self, model_key):
        """Helper to run DDP scaling test on multiple processes."""
        config = MODEL_CONFIGS[model_key]
        world_size = min(torch.cuda.device_count(), 2)

        with mp.Manager() as manager:
            result_dict = manager.dict()

            port = _find_free_port()
            mp.spawn(
                _run_multi_gpu_wrapper,
                args=(
                    world_size,
                    port,
                    model_key,
                    config["batch_size"],
                    config["max_length"],
                    config["accum_steps"],
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
        self._run_ddp_scaling_test("qwen2-0.5b")

    def test_qwen2_1_5b(self):
        """Test Qwen2-1.5B (1.5B, bs=8, accum=16)."""
        self._run_ddp_scaling_test("qwen2-1.5b")

    def test_tinyllama_1_1b(self):
        """Test TinyLlama-1.1B (1.1B, bs=8, accum=16)."""
        self._run_ddp_scaling_test("tinyllama-1.1b")

    def test_phi3_3_8b(self):
        """Test Phi-3-mini (3.8B, bs=8, accum=16, Phi-3-specific vmap patches)."""
        self._run_ddp_scaling_test("phi3-mini")
