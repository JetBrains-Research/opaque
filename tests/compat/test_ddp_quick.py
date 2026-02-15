# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Quick DDP test with minimal config to identify issues faster."""

import os

import pytest
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from opaque import clipped_grad, make_functional
from opaque.utils.pytree import tree_map


def _is_distributed():
    return dist.is_available() and dist.is_initialized()


def _rank():
    if _is_distributed():
        return dist.get_rank()
    return 0


def _build_text_batch(batch_size):
    base_texts = [
        "Hello world test",
        "Another example",
        "Third sample",
        "Final one",
        "Short prompt",
        "Yet another example",
        "Testing a longer-ish input",
        "Final sample in batch",
    ]

    if batch_size <= len(base_texts):
        return base_texts[:batch_size]

    repeats = (batch_size + len(base_texts) - 1) // len(base_texts)
    return (base_texts * repeats)[:batch_size]


def _has_min_gpu_memory(min_gb=8):
    if not torch.cuda.is_available():
        return False
    try:
        total_bytes = torch.cuda.get_device_properties(0).total_memory
    except Exception:
        return False
    return total_bytes >= min_gb * (1024**3)


@pytest.mark.compat
@pytest.mark.slow
@pytest.mark.gpu
class TestDDPQuickSanity:
    """Quick DDP sanity check - minimal config."""

    def test_ddp_qwen2_0_5b_basic(self):
        """Quick DDP test with Qwen2-0.5B, batch=4, accum=8, 2 training steps."""
        if not _is_distributed():
            pytest.skip("Requires DDP (torch.distributed initialized)")

        if not _has_min_gpu_memory(12):
            pytest.skip("Requires CUDA GPU with >= 12GB memory")

        model_name = "Qwen/Qwen2-0.5B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        config = AutoConfig.from_pretrained(model_name)
        config._attn_implementation = "eager"
        config.num_hidden_layers = 2  # Reduce layers for speed

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float16,
        )

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        batch_size = 4
        max_length = 512
        accum_steps = 8
        training_steps = 2
        lr = 1e-3

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

        def per_example_loss(
            trainable_params, frozen_params, input_ids_single, mask_single, labels_single
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

        for step in range(training_steps):
            accumulated = None
            for accum in range(accum_steps):
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
            accumulated = tree_map(lambda x: x * scale, accumulated)

            trainable_params = tree_map(
                lambda param, grad: param - lr * grad,
                trainable_params,
                accumulated,
            )

        # Synchronize across ranks
        if _is_distributed():
            dist.barrier()

        assert trainable_params is not None
