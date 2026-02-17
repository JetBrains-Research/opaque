# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Single-GPU validation tests for real HuggingFace models with LoRA + DP-SGD.

Tests use shared model configs and utilities from tests/conftest.py to avoid duplication.
"""

import pytest

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from tests.conftest import (
    MODEL_CONFIGS,
    has_min_gpu_memory,
    load_model_with_lora,
    run_dp_training_step,
)


@pytest.mark.test
@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.integration
class TestRealModelsSingleGPU:
    """Single-GPU validation with real pretrained weights using shared utilities."""

    @pytest.mark.parametrize(
        "model_key",
        ["qwen2-0.5b", "qwen2-1.5b", "tinyllama-1.1b", "phi3-mini"],
        ids=lambda k: MODEL_CONFIGS[k]["model_id"],
    )
    def test_model_lora_dp_training(self, model_key):
        """Run DP-SGD training with LoRA using shared test utilities."""
        config = MODEL_CONFIGS[model_key]

        if not has_min_gpu_memory(config["min_mem_gb"]):
            pytest.skip(f"Requires CUDA GPU with >= {config['min_mem_gb']}GB memory")

        # Load model using shared utility
        model, tokenizer = load_model_with_lora(config, device="cuda")

        # Run training using shared utility
        grads, state = run_dp_training_step(
            model,
            tokenizer,
            batch_size=config["batch_size"],
            max_length=config["max_length"],
            accum_steps=config["accum_steps"],
            training_steps=3,
            learning_rate=1e-3,
            l2_clip_norm=1.0,
        )

        # Verify training produced gradients
        assert len(grads) > 0
