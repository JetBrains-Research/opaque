"""Conftest for the opaque meta-package tests (validation + cross-mechanism).

Shared GPU/device fixtures come from the workspace-root conftest.py.
This file adds HF model loading helpers used by validation tests.
"""

import os

import pytest
import torch

from conftest import get_default_gpu_device


# =============================================================================
# Model Testing Utilities
# =============================================================================

STANDARD_LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
}

MODEL_CONFIGS = {
    "qwen2-0.5b": {
        "model_id": "Qwen/Qwen2-0.5B",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "min_mem_gb": 12,
        "max_length": 2048,
        "trust_remote_code": False,
        "batch_size": 8,
        "accum_steps": 32,
    },
    "qwen2-1.5b": {
        "model_id": "Qwen/Qwen2-1.5B",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "min_mem_gb": 12,
        "max_length": 1024,
        "trust_remote_code": False,
        "batch_size": 8,
        "accum_steps": 16,
    },
    "tinyllama-1.1b": {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "min_mem_gb": 12,
        "max_length": 1024,
        "trust_remote_code": False,
        "batch_size": 8,
        "accum_steps": 16,
    },
    "phi3-mini": {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "target_modules": ["qkv_proj"],
        "min_mem_gb": 16,
        "max_length": 2048,
        "trust_remote_code": True,
        "batch_size": 8,
        "accum_steps": 16,
    },
}


def build_text_batch(batch_size):
    """Create deterministic text batch for testing."""
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


def has_min_gpu_memory(min_gb, device=None):
    """Check if GPU has minimum required memory."""
    if device is None:
        gpu_device = get_default_gpu_device()
    else:
        gpu_device = torch.device(device)

    if gpu_device is None:
        return False

    required_bytes = int(min_gb * (1024**3))

    if gpu_device.type == "cuda":
        try:
            free_bytes, _total_from_driver = torch.cuda.mem_get_info(0)
            if free_bytes > 0:
                return free_bytes >= required_bytes
        except Exception:
            pass
        try:
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            return total_bytes >= required_bytes
        except Exception:
            return False

    if gpu_device.type == "mps":
        try:
            recommended_bytes = None
            allocated_bytes = None

            if hasattr(torch.mps, "recommended_max_memory"):
                candidate = int(torch.mps.recommended_max_memory())
                if candidate > 0:
                    recommended_bytes = candidate

            if hasattr(torch.mps, "current_allocated_memory"):
                candidate = int(torch.mps.current_allocated_memory())
                if candidate >= 0:
                    allocated_bytes = candidate

            if recommended_bytes is not None and allocated_bytes is not None:
                free_bytes = max(0, recommended_bytes - allocated_bytes)
                return free_bytes >= required_bytes

            if recommended_bytes is not None:
                return recommended_bytes >= required_bytes
        except Exception:
            return False
        return False

    return False


def gpu_memory_gate_reason(min_gb, device=None):
    """Return standardized skip reason for GPU memory gating."""
    if device is None:
        gpu_device = get_default_gpu_device()
    else:
        gpu_device = torch.device(device)

    if gpu_device is None:
        return f"Requires GPU with >= {min_gb}GB memory"

    if gpu_device.type == "cuda":
        try:
            free_bytes, _total_from_driver = torch.cuda.mem_get_info(0)
            free_gb = free_bytes / (1024**3)
            return (
                f"Requires >= {min_gb}GB free CUDA memory "
                f"(currently {free_gb:.2f}GB free)"
            )
        except Exception:
            return f"Requires CUDA GPU with >= {min_gb}GB memory"

    if gpu_device.type == "mps":
        try:
            recommended = None
            allocated = None
            if hasattr(torch.mps, "recommended_max_memory"):
                recommended = int(torch.mps.recommended_max_memory())
            if hasattr(torch.mps, "current_allocated_memory"):
                allocated = int(torch.mps.current_allocated_memory())

            if (
                recommended
                and recommended > 0
                and allocated is not None
                and allocated >= 0
            ):
                free_gb = max(0, recommended - allocated) / (1024**3)
                return (
                    f"Requires >= {min_gb}GB free MPS memory "
                    f"(estimated {free_gb:.2f}GB free)"
                )

            if recommended and recommended > 0:
                recommended_gb = recommended / (1024**3)
                return (
                    f"Requires >= {min_gb}GB MPS recommended memory "
                    f"(currently {recommended_gb:.2f}GB)"
                )
        except Exception:
            pass

        return f"Requires MPS memory introspection and >= {min_gb}GB available"

    return f"Requires GPU with >= {min_gb}GB memory"


def load_model_with_lora(
    model_config, device="cuda", dtype=torch.float16, lora_config=None
):
    """Load HuggingFace model with LoRA adapters."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_id = model_config["model_id"]
    trust_remote_code = model_config["trust_remote_code"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    ).to(device)

    if lora_config is None:
        lora_config = STANDARD_LORA_CONFIG.copy()

    lora_config["target_modules"] = model_config["target_modules"]
    model = get_peft_model(model, LoraConfig(**lora_config))

    return model, tokenizer


def run_dp_training_step(
    model,
    tokenizer,
    batch_size,
    max_length,
    accum_steps,
    training_steps=3,
    learning_rate=1e-3,
    clipping_norm=1.0,
):
    """Run DP-SGD training with clipped gradients and gradient accumulation."""
    from opaque.clipping import clipped_grad
    from opaque.utils import make_functional
    from opaque.utils.pytree import tree_map

    device = next(model.parameters()).device

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
        trainable_params, frozen_params, ids_single, mask_single, labels_single
    ):
        all_params = {**frozen_params, **trainable_params}
        outputs = fmodel(
            all_params, ids_single, attention_mask=mask_single, labels=labels_single
        )
        return outputs.loss

    grad_fn, clip_state = clipped_grad(
        per_example_loss,
        argnums=0,
        batch_argnums=(2, 3, 4),
        clipping_norm=clipping_norm,
    )

    state = clip_state
    trainable_params = trainable
    last_accumulated = None

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

        trainable_params = tree_map(
            lambda p, g: p - learning_rate * g,
            trainable_params,
            accumulated,
        )

        last_accumulated = accumulated

    return last_accumulated, state


@pytest.fixture(scope="session")
def model_configs():
    """Provide model configurations to tests."""
    return MODEL_CONFIGS


@pytest.fixture(scope="session")
def standard_lora_config():
    """Provide standard LoRA configuration."""
    return STANDARD_LORA_CONFIG
