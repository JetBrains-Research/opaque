# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Slow tests that load real Hugging Face models with realistic batch sizes."""

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from opaque import clipped_grad, make_functional
from opaque.utils.pytree import tree_map


# Memory-aware configs for A100 40GB
REAL_MODELS = [
    {
        "name": "Qwen/Qwen2-0.5B",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "min_mem_gb": 12,
        "max_length": 2048,
        "trust_remote_code": False,
        "batch_sizes": [8],
        "accum_steps": [32],
    },
    {
        "name": "Qwen/Qwen2-1.5B",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "min_mem_gb": 12,
        "max_length": 1024,
        "trust_remote_code": False,
        "batch_sizes": [8],
        "accum_steps": [16],
    },
    {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "min_mem_gb": 12,
        "max_length": 1024,
        "trust_remote_code": False,
        "batch_sizes": [8],
        "accum_steps": [16],
    },
]
REALISTIC_BATCH_SIZES = (8, 16)  # Unused, using per-model batch_sizes
REALISTIC_ACCUM_STEPS = (16, 32)  # Unused, using per-model accum_steps
TRAINING_STEPS = 3
LEARNING_RATE = 1e-3


def _has_min_gpu_memory(min_gb):
    if not torch.cuda.is_available():
        return False
    try:
        total_bytes = torch.cuda.get_device_properties(0).total_memory
    except Exception:
        return False
    return total_bytes >= min_gb * (1024**3)


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


def _load_model_and_tokenizer(model_spec):
    model_name = model_spec["name"]
    trust_remote_code = model_spec["trust_remote_code"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    config._attn_implementation = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float16,
        trust_remote_code=trust_remote_code,
    ).to("cuda")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=model_spec["target_modules"],
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def _run_clipped_grad_with_accum(model, tokenizer, batch_size, max_length, accum_steps):
    if max_length < 1024:
        raise ValueError("max_length must be >= 1024")

    device = next(model.parameters()).device

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
    last_accumulated = None

    for _ in range(TRAINING_STEPS):
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
        accumulated = tree_map(lambda x: x * scale, accumulated)
        trainable_params = tree_map(
            lambda param, grad: param - LEARNING_RATE * grad,
            trainable_params,
            accumulated,
        )
        last_accumulated = accumulated

    return last_accumulated, state


@pytest.mark.compat
@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.integration
class TestRealModelsSingleGPU:
    """Single-GPU validation with real pretrained weights."""

    @pytest.mark.parametrize("model_spec", REAL_MODELS, ids=lambda s: s["name"])
    def test_real_models_lora_memory_aware(self, model_spec):
        """Run clipped_grad with memory-aware batch size and accumulation."""
        if not _has_min_gpu_memory(model_spec["min_mem_gb"]):
            pytest.skip(
                f"Requires CUDA GPU with >= {model_spec['min_mem_gb']}GB memory"
            )

        model, tokenizer = _load_model_and_tokenizer(model_spec)
        
        # Test all batch_size/accum_steps combos for this model
        for batch_size in model_spec["batch_sizes"]:
            for accum_steps in model_spec["accum_steps"]:
                grads, _ = _run_clipped_grad_with_accum(
                    model,
                    tokenizer,
                    batch_size=batch_size,
                    max_length=model_spec["max_length"],
                    accum_steps=accum_steps,
                )
                assert len(grads) > 0
