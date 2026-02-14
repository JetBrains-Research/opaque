# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Test different attention implementations with vmap/clipped_grad.

Tests eager, SDPA, Flash Attention 2, and flex_attention.
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

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Flash Attention requires CUDA"
    )
    def test_flash_attention_2(self, qwen2_config, qwen2_tokenizer):
        """Test Flash Attention 2 - should fail due to torch.nonzero incompatibility."""
        pytest.importorskip("flash_attn")

        if not torch.cuda.is_available():
            pytest.skip("Flash Attention 2 requires CUDA")

        qwen2_config._attn_implementation = "flash_attention_2"
        model = prepare_lora_model(qwen2_config).to("cuda")

        # Flash Attention 2 uses torch.nonzero which has dynamic output shape
        # and is incompatible with vmap
        with pytest.raises(RuntimeError, match="vmap.*nonzero"):
            grads, _ = run_clipped_grad_test(model, qwen2_tokenizer)

    def test_flex_attention(self, qwen2_config, qwen2_tokenizer, device):
        """Test flex_attention - currently incompatible with vmap. Works on CPU and CUDA."""
        # flex_attention has vmap compatibility issues with tensor metadata
        try:
            qwen2_config._attn_implementation = "flex_attention"
        except Exception:
            pytest.skip("transformers does not support flex_attention yet")

        model = prepare_lora_model(qwen2_config).to(device)

        # flex_attention fails with metadata assertion errors under vmap
        with pytest.raises(AssertionError, match="False != True"):
            grads, _ = run_clipped_grad_test(model, qwen2_tokenizer)
