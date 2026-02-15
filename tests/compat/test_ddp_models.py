# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""DDP tests for real Hugging Face models with clipped gradients.

Follows Opacus testing pattern with mp.spawn() for multi-GPU tests.

Run with:
  pytest tests/compat/test_ddp_real_models.py -v -s

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


def _run_ddp_training_steps(
    rank, world_size, model_name, target_modules, max_length, batch_size, result_dict
):
    """Run training with clipped_grad on multiple GPUs.

    Args:
        rank: Process rank
        world_size: Total number of processes
        model_name: HF model ID
        target_modules: LoRA target modules
        max_length: Tokenizer max length
        batch_size: Batch size per GPU
        result_dict: Shared dict to store success flag
    """
    torch.manual_seed(42)
    torch.cuda.set_device(rank)

    try:
        _setup_ddp(rank, world_size)
        device = torch.device(f"cuda:{rank}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
        config._attn_implementation = "eager"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            dtype=torch.float16,
        ).to(device)

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)

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
            l2_clip_norm=1.0,
        )

        state = clip_state
        trainable_params = trainable
        learning_rate = 1e-3

        for step in range(2):
            accumulated = None
            for _ in range(4):
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

            scale = 0.25
            accumulated = tree_map(lambda x: x * scale, accumulated)

            trainable_params = tree_map(
                lambda p, g: p - learning_rate * g,
                trainable_params,
                accumulated,
            )

        result_dict[rank] = True

    except Exception as e:
        result_dict[rank] = False
        if rank == 0:
            print(f"Error on rank {rank}: {e}")
        raise

    finally:
        _cleanup_ddp()


def _run_multi_gpu_demo(
    rank, world_size, model_name, target_modules, max_length, batch_size, result_dict
):
    """Wrapper for mp.spawn."""
    _run_ddp_training_steps(
        rank, world_size, model_name, target_modules, max_length, batch_size, result_dict
    )


@unittest.skipIf(
    torch.cuda.device_count() < 2,
    "Need at least 2 GPUs for DDP tests",
)
class TestDDPRealModels(unittest.TestCase):
    """DDP tests with real HF models using mp.spawn pattern (Opacus style)."""

    def _run_ddp_test(self, model_name, target_modules, max_length, batch_size):
        """Helper to run DDP test on multiple processes."""
        world_size = min(torch.cuda.device_count(), 2)

        with mp.Manager() as manager:
            result_dict = manager.dict()

            mp.spawn(
                _run_multi_gpu_demo,
                args=(world_size, model_name, target_modules, max_length, batch_size, result_dict),
                nprocs=world_size,
                join=True,
            )

            for rank in range(world_size):
                self.assertTrue(
                    result_dict[rank],
                    f"Training failed on rank {rank}",
                )

    def test_qwen2_0_5b_ddp(self):
        """Test Qwen2-0.5B with DDP."""
        self._run_ddp_test(
            model_name="Qwen/Qwen2-0.5B",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            max_length=1024,
            batch_size=4,
        )

    def test_pythia_160m_ddp(self):
        """Test Pythia-160M with DDP."""
        self._run_ddp_test(
            model_name="EleutherAI/pythia-160m",
            target_modules=["query_key_value", "dense"],
            max_length=1024,
            batch_size=4,
        )

    def test_gpt2_ddp(self):
        """Test GPT-2 with DDP."""
        self._run_ddp_test(
            model_name="gpt2",
            target_modules=["c_attn", "c_proj"],
            max_length=1024,
            batch_size=4,
        )
