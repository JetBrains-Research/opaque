# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Test different attention implementations with vmap/clipped_grad.

Tests eager and SDPA attention implementations that are compatible with vmap.

Known incompatibilities (not tested):
- Flash Attention 2: Uses torch.nonzero which has dynamic output shape
- flex_attention: Has tensor metadata issues under vmap
"""

import pytest
import torch

from tests.compat.conftest import prepare_lora_model, run_clipped_grad_test


class TestAttentionImplementations:
    """Test different attention implementations work with clipped_grad."""

    def test_eager_attention(self, qwen2_config, qwen2_tokenizer, device):
        """Test eager attention (explicitly patched). Works on CPU and CUDA."""
        qwen2_config._attn_implementation = "eager"
        model = prepare_lora_model(qwen2_config).to(device)
        grads, _ = run_clipped_grad_test(model, qwen2_tokenizer)
        assert len(grads) > 0

    def test_sdpa_attention(self, qwen2_config, qwen2_tokenizer, device):
        """Test SDPA attention (default, uses patched repeat_kv). Works on CPU and CUDA."""
        qwen2_config._attn_implementation = "sdpa"
        model = prepare_lora_model(qwen2_config).to(device)
        grads, _ = run_clipped_grad_test(model, qwen2_tokenizer)
        assert len(grads) > 0
