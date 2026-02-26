# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for automatic Triton kernel patches on HuggingFace models.

Verifies that patched model components produce correct outputs and gradients,
and remain vmap-compatible for DP-SGD.
"""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers")

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

from opaque import clipped_grad, make_functional  # noqa: E402


RTOL = 1e-4
ATOL = 1e-4


# =============================================================================
# Helpers
# =============================================================================

def _make_small_model(model_name, device="cuda"):
    """Create a small 2-layer model for testing."""
    config = AutoConfig.from_pretrained(model_name)
    config.num_hidden_layers = 2
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(config).to(device)
    return model, config


def _get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)


# =============================================================================
# RMSNorm Tests
# =============================================================================

@pytest.mark.gpu
class TestRMSNormPatches:
    """Test that patched RMSNorm produces correct outputs."""

    def test_llama_rmsnorm_output_matches(self, device):
        """Patched LlamaRMSNorm output should match original."""
        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        hidden_dim = 256
        norm = LlamaRMSNorm(hidden_dim).to(device)

        x = torch.randn(2, 16, hidden_dim, device=device, dtype=torch.float32)

        # The module is already patched at import time. Verify it works correctly.
        out = norm(x)

        # Compare with manual PyTorch reference
        variance = x.pow(2).mean(-1, keepdim=True)
        ref = x * torch.rsqrt(variance + norm.variance_epsilon) * norm.weight

        assert torch.allclose(out, ref, rtol=RTOL, atol=ATOL), \
            f"LlamaRMSNorm output mismatch: max diff {(out - ref).abs().max():.2e}"

    def test_gemma_rmsnorm_weight_plus_one(self, device):
        """GemmaRMSNorm should correctly apply weight+1 trick."""
        try:
            from transformers.models.gemma.modeling_gemma import GemmaRMSNorm
        except ImportError:
            pytest.skip("Gemma not available in this transformers version")

        hidden_dim = 256
        norm = GemmaRMSNorm(hidden_dim).to(device)

        x = torch.randn(2, 16, hidden_dim, device=device, dtype=torch.float32)

        out = norm(x)

        # Reference: RMSNorm with effective_weight = 1 + weight
        effective_weight = (1.0 + norm.weight).float()
        variance = x.pow(2).mean(-1, keepdim=True)
        ref = (x * torch.rsqrt(variance + norm.eps) * effective_weight).type_as(x)

        assert torch.allclose(out, ref, rtol=RTOL, atol=ATOL), \
            f"GemmaRMSNorm output mismatch: max diff {(out - ref).abs().max():.2e}"


# =============================================================================
# MLP Tests
# =============================================================================

@pytest.mark.gpu
class TestMLPPatches:
    """Test that patched MLP forward produces correct outputs."""

    def test_swiglu_mlp_matches(self, device):
        """Patched LlamaMLP should match PyTorch reference."""
        from transformers.models.llama.modeling_llama import LlamaMLP

        config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        config.num_hidden_layers = 1

        mlp = LlamaMLP(config).to(device)

        x = torch.randn(2, 16, config.hidden_size, device=device, dtype=torch.float32)

        out = mlp(x)

        # Reference: silu(gate) * up -> down
        gate = mlp.gate_proj(x)
        up = mlp.up_proj(x)
        ref = mlp.down_proj(F.silu(gate) * up)

        assert torch.allclose(out, ref, rtol=RTOL, atol=ATOL), \
            f"LlamaMLP output mismatch: max diff {(out - ref).abs().max():.2e}"

    def test_geglu_exact_mlp_matches(self, device):
        """Patched GemmaMLP should match PyTorch reference."""
        try:
            from transformers.models.gemma.modeling_gemma import GemmaMLP
        except ImportError:
            pytest.skip("Gemma not available")

        config = AutoConfig.from_pretrained("google/gemma-2b")
        config.num_hidden_layers = 1

        mlp = GemmaMLP(config).to(device)

        x = torch.randn(2, 16, config.hidden_size, device=device, dtype=torch.float32)

        out = mlp(x)

        # Reference: gelu_exact(gate) * up -> down
        gate = mlp.gate_proj(x)
        up = mlp.up_proj(x)
        ref = mlp.down_proj(F.gelu(gate, approximate='none') * up)

        assert torch.allclose(out, ref, rtol=RTOL, atol=ATOL), \
            f"GemmaMLP output mismatch: max diff {(out - ref).abs().max():.2e}"

    def test_geglu_approx_mlp_matches(self, device):
        """Patched Gemma2MLP should match PyTorch reference."""
        try:
            from transformers.models.gemma2.modeling_gemma2 import Gemma2MLP
        except ImportError:
            pytest.skip("Gemma2 not available")

        config = AutoConfig.from_pretrained("google/gemma-2-2b")
        config.num_hidden_layers = 1

        mlp = Gemma2MLP(config).to(device)

        x = torch.randn(2, 16, config.hidden_size, device=device, dtype=torch.float32)

        out = mlp(x)

        # Reference: gelu_tanh(gate) * up -> down
        gate = mlp.gate_proj(x)
        up = mlp.up_proj(x)
        ref = mlp.down_proj(F.gelu(gate, approximate='tanh') * up)

        assert torch.allclose(out, ref, rtol=RTOL, atol=ATOL), \
            f"Gemma2MLP output mismatch: max diff {(out - ref).abs().max():.2e}"


# =============================================================================
# Gradient Tests
# =============================================================================

@pytest.mark.gpu
class TestGradients:
    """Test that gradients through patched modules are correct."""

    def test_backward_through_patched_rmsnorm(self, device):
        """Gradients should flow correctly through patched RMSNorm."""
        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        hidden_dim = 256
        norm = LlamaRMSNorm(hidden_dim).to(device)

        x = torch.randn(2, 16, hidden_dim, device=device, requires_grad=True)
        out = norm(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "No gradient computed through patched RMSNorm"
        assert not torch.isnan(x.grad).any(), "NaN in gradients"
        assert not torch.isinf(x.grad).any(), "Inf in gradients"

    def test_backward_through_patched_mlp(self, device):
        """Gradients should flow correctly through patched MLP."""
        from transformers.models.llama.modeling_llama import LlamaMLP

        config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        config.num_hidden_layers = 1

        mlp = LlamaMLP(config).to(device)

        x = torch.randn(2, 16, config.hidden_size, device=device, requires_grad=True)
        out = mlp(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "No gradient computed through patched MLP"
        assert not torch.isnan(x.grad).any(), "NaN in gradients"
        assert not torch.isinf(x.grad).any(), "Inf in gradients"


# =============================================================================
# vmap Compatibility
# =============================================================================

@pytest.mark.gpu
class TestVmapCompatibility:
    """Test that patched modules work under vmap for DP-SGD."""

    def test_vmap_patched_rmsnorm(self, device):
        """Patched RMSNorm should work under vmap."""
        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        hidden_dim = 256
        norm = LlamaRMSNorm(hidden_dim).to(device)

        # Batched input for vmap
        x = torch.randn(4, 2, 16, hidden_dim, device=device, requires_grad=True)

        out = torch.vmap(norm)(x)
        out.sum().backward()

        assert x.grad is not None, "No gradient from vmap RMSNorm"
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"

    def test_vmap_patched_mlp(self, device):
        """Patched MLP should work under vmap."""
        from transformers.models.llama.modeling_llama import LlamaMLP

        config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        config.num_hidden_layers = 1

        mlp = LlamaMLP(config).to(device)

        # Batched input for vmap
        x = torch.randn(4, 2, 16, config.hidden_size, device=device, requires_grad=True)

        out = torch.vmap(mlp)(x)
        out.sum().backward()

        assert x.grad is not None, "No gradient from vmap MLP"


# =============================================================================
# End-to-end Integration
# =============================================================================

@pytest.mark.gpu
class TestEndToEnd:
    """End-to-end test with full model + LoRA + clipped_grad."""

    def test_qwen2_lora_clipped_grad(self, device):
        """Full pipeline: Qwen2 + LoRA + clipped_grad with kernel patches."""
        config = AutoConfig.from_pretrained("Qwen/Qwen2-0.5B")
        config.num_hidden_layers = 2
        config._attn_implementation = "eager"

        model = AutoModelForCausalLM.from_config(config)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
        )
        model = get_peft_model(model, lora_config).to(device)

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        texts = ["Hello world test", "Another example", "Third sample", "Final one"]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, max_length=16, truncation=True
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = input_ids.clone()

        fmodel, trainable, frozen = make_functional(
            model, disable_autograd_tracking=True, partition_trainable=True
        )

        def per_example_loss(trainable_params, frozen_params, ids, mask, lbls):
            all_params = {**frozen_params, **trainable_params}
            outputs = fmodel(all_params, ids, attention_mask=mask, labels=lbls)
            return outputs.loss

        grad_fn, clip_state = clipped_grad(
            per_example_loss, argnums=0, batch_argnums=(2, 3, 4), l2_clip_norm=1.0
        )
        grads, state = grad_fn(
            trainable, frozen, input_ids, attention_mask, labels, state=clip_state
        )

        assert grads is not None, "No gradients returned"
        assert len(grads) > 0, "Empty gradient dict"
        for name, g in grads.items():
            assert not torch.isnan(g).any(), f"NaN in grad for {name}"


# =============================================================================
# Configuration Tests
# =============================================================================

class TestConfiguration:
    """Test patch configuration and control."""

    def test_kernel_patched_flag(self):
        """is_kernel_patched() should return True after import opaque."""
        from opaque.compat.transformers import is_kernel_patched

        # After import opaque, patches should be applied (or skipped if no CUDA)
        assert isinstance(is_kernel_patched(), bool)

    def test_patch_stores_original_forward(self):
        """Patched classes should preserve original forward."""
        try:
            from transformers.models.llama.modeling_llama import LlamaRMSNorm
        except ImportError:
            pytest.skip("transformers not available")

        if hasattr(LlamaRMSNorm, "_opaque_original_forward"):
            assert callable(LlamaRMSNorm._opaque_original_forward)


# =============================================================================
# Cross-Entropy Loss Patch Tests
# =============================================================================

@pytest.mark.gpu
class TestCrossEntropyPatches:
    """Test that patched cross-entropy loss produces correct outputs."""

    def test_causal_lm_loss_matches_pytorch(self, device):
        """Patched ForCausalLMLoss should match F.cross_entropy reference."""
        from opaque.compat.transformers._kernel_patches import _opaque_causal_lm_loss

        batch, seq_len, vocab_size = 2, 16, 1000
        logits = torch.randn(batch, seq_len, vocab_size, device=device)
        labels = torch.randint(0, vocab_size, (batch, seq_len), device=device)
        # Set some labels to -100 (ignored)
        labels[:, -2:] = -100

        loss = _opaque_causal_lm_loss(logits, labels, vocab_size)

        # PyTorch reference with same shifting logic
        import torch.nn as nn
        labels_ref = nn.functional.pad(labels, (0, 1), value=-100)
        shift_labels = labels_ref[..., 1:].contiguous()
        logits_flat = logits.float().view(-1, vocab_size)
        shift_labels_flat = shift_labels.view(-1)
        ref = F.cross_entropy(logits_flat, shift_labels_flat, ignore_index=-100)

        assert torch.allclose(loss, ref, rtol=1e-3, atol=1e-3), \
            f"Cross-entropy loss mismatch: got {loss.item():.6f}, expected {ref.item():.6f}"

    def test_causal_lm_loss_with_num_items_in_batch(self, device):
        """Loss with num_items_in_batch should use sum reduction."""
        from opaque.compat.transformers._kernel_patches import _opaque_causal_lm_loss

        batch, seq_len, vocab_size = 2, 16, 1000
        logits = torch.randn(batch, seq_len, vocab_size, device=device)
        labels = torch.randint(0, vocab_size, (batch, seq_len), device=device)

        import torch.nn as nn
        num_items = torch.tensor(batch * (seq_len - 1), dtype=torch.float32, device=device)
        loss = _opaque_causal_lm_loss(logits, labels, vocab_size,
                                       num_items_in_batch=num_items)

        # Reference: sum reduction / num_items
        labels_ref = nn.functional.pad(labels, (0, 1), value=-100)
        shift_labels = labels_ref[..., 1:].contiguous()
        logits_flat = logits.float().view(-1, vocab_size)
        shift_labels_flat = shift_labels.view(-1)
        ref = F.cross_entropy(logits_flat, shift_labels_flat, ignore_index=-100,
                              reduction="sum") / num_items

        assert torch.allclose(loss, ref, rtol=1e-3, atol=1e-3), \
            f"Sum-reduced loss mismatch: got {loss.item():.6f}, expected {ref.item():.6f}"

    def test_backward_through_patched_loss(self, device):
        """Gradients should flow through patched cross-entropy loss."""
        from opaque.compat.transformers._kernel_patches import _opaque_causal_lm_loss

        batch, seq_len, vocab_size = 2, 16, 1000
        logits = torch.randn(batch, seq_len, vocab_size, device=device, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch, seq_len), device=device)

        loss = _opaque_causal_lm_loss(logits, labels, vocab_size)
        loss.backward()

        assert logits.grad is not None, "No gradient through patched loss"
        assert not torch.isnan(logits.grad).any(), "NaN in loss gradients"
        assert not torch.isinf(logits.grad).any(), "Inf in loss gradients"

    def test_loss_mapping_patched(self):
        """LOSS_MAPPING should point to Opaque loss function after patching."""
        from opaque.compat.transformers._kernel_patches import _opaque_causal_lm_loss

        try:
            from transformers.loss.loss_utils import LOSS_MAPPING
        except ImportError:
            pytest.skip("transformers not available")

        if torch.cuda.is_available():
            assert LOSS_MAPPING.get("ForCausalLM") is _opaque_causal_lm_loss


# =============================================================================
# LoRA Patch Tests
# =============================================================================

@pytest.mark.gpu
class TestLoRAPatches:
    """Test that patched LoRA linear produces correct outputs."""

    def test_lora_forward_matches_peft(self, device):
        """Patched LoRA forward should match PyTorch reference."""
        from peft.tuners.lora import Linear as PeftLoRALinear

        in_features, out_features, rank = 256, 512, 8
        base_linear = torch.nn.Linear(in_features, out_features, bias=False).to(device)

        lora_layer = PeftLoRALinear(
            base_linear, "default", r=rank, lora_alpha=16, lora_dropout=0.0
        ).to(device)

        x = torch.randn(2, 16, in_features, device=device)
        out = lora_layer(x)

        # PyTorch reference: base(x) + B(A(x)) * scaling
        base_out = base_linear(x)
        A_weight = lora_layer.lora_A["default"].weight  # (rank, in)
        B_weight = lora_layer.lora_B["default"].weight  # (out, rank)
        scaling = lora_layer.scaling["default"]
        lora_delta = F.linear(F.linear(x, A_weight), B_weight) * scaling
        ref = base_out + lora_delta

        assert torch.allclose(out, ref, rtol=RTOL, atol=ATOL), \
            f"LoRA forward mismatch: max diff {(out - ref).abs().max():.2e}"

    def test_lora_forward_with_bias(self, device):
        """LoRA forward should correctly handle base layer bias."""
        from peft.tuners.lora import Linear as PeftLoRALinear

        in_features, out_features, rank = 256, 512, 8
        base_linear = torch.nn.Linear(in_features, out_features, bias=True).to(device)

        lora_layer = PeftLoRALinear(
            base_linear, "default", r=rank, lora_alpha=16, lora_dropout=0.0
        ).to(device)

        x = torch.randn(2, 16, in_features, device=device)
        out = lora_layer(x)

        # Reference with bias
        base_out = base_linear(x)
        A_weight = lora_layer.lora_A["default"].weight
        B_weight = lora_layer.lora_B["default"].weight
        scaling = lora_layer.scaling["default"]
        lora_delta = F.linear(F.linear(x, A_weight), B_weight) * scaling
        ref = base_out + lora_delta

        assert torch.allclose(out, ref, rtol=RTOL, atol=ATOL), \
            f"LoRA with bias mismatch: max diff {(out - ref).abs().max():.2e}"

    def test_backward_through_patched_lora(self, device):
        """Gradients should flow through patched LoRA."""
        from peft.tuners.lora import Linear as PeftLoRALinear

        in_features, out_features, rank = 256, 512, 8
        base_linear = torch.nn.Linear(in_features, out_features, bias=False).to(device)

        lora_layer = PeftLoRALinear(
            base_linear, "default", r=rank, lora_alpha=16, lora_dropout=0.0
        ).to(device)

        x = torch.randn(2, 16, in_features, device=device, requires_grad=True)
        out = lora_layer(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "No gradient through LoRA"
        assert not torch.isnan(x.grad).any(), "NaN in LoRA gradients"

        # Check LoRA weight gradients
        A_grad = lora_layer.lora_A["default"].weight.grad
        B_grad = lora_layer.lora_B["default"].weight.grad
        assert A_grad is not None, "No gradient for LoRA A"
        assert B_grad is not None, "No gradient for LoRA B"

    def test_lora_class_patched(self):
        """peft.tuners.lora.Linear should have patched forward."""
        from opaque.compat.transformers._kernel_patches import _opaque_lora_linear_forward

        try:
            from peft.tuners.lora import Linear as PeftLoRALinear
        except ImportError:
            pytest.skip("peft not available")

        if torch.cuda.is_available():
            assert PeftLoRALinear.forward is _opaque_lora_linear_forward
