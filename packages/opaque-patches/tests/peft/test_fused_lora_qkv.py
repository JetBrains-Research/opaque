import pytest
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2Config,
    Qwen3Config,
)

from opaque.api.engine.clipping import clipped_grad
from opaque.patches import apply_model_patches, apply_runtime_patches
from opaque.torch.functional import make_functional

from .._helpers import requires_hf_auth

apply_runtime_patches()
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Kernel patch compatibility tests require CUDA/Triton",
)

RTOL = 0.0001
ATOL = 0.0001


@pytest.mark.cuda
class TestFusedLoRAQKV:
    """Test fused LoRA QKV patching via Opaque_LoRA_QKV kernel."""

    def test_fused_lora_qkv_forward(self, device):
        """Fused LoRA QKV forward should match PyTorch matmul reference."""
        from opaque.api.patches.kernels.lora import Opaque_LoRA_QKV

        torch.manual_seed(42)
        batch, seq, hidden, q_out, kv_out, rank = (2, 16, 256, 256, 64, 8)
        scaling = 2.0
        x = torch.randn(batch, seq, hidden, device=device, dtype=torch.float32)
        Wq = torch.randn(q_out, hidden, device=device)
        Wk = torch.randn(kv_out, hidden, device=device)
        Wv = torch.randn(kv_out, hidden, device=device)
        Aq = torch.randn(hidden, rank, device=device)
        Bq = torch.randn(rank, q_out, device=device)
        Ak = torch.randn(hidden, rank, device=device)
        Bk = torch.randn(rank, kv_out, device=device)
        Av = torch.randn(hidden, rank, device=device)
        Bv = torch.randn(rank, kv_out, device=device)
        Q, K, V = Opaque_LoRA_QKV.apply(
            x, Wq, Aq, Bq, scaling, Wk, Ak, Bk, scaling, Wv, Av, Bv, scaling
        )
        ref_q = F.linear(x, Wq) + x @ Aq @ Bq * scaling
        ref_k = F.linear(x, Wk) + x @ Ak @ Bk * scaling
        ref_v = F.linear(x, Wv) + x @ Av @ Bv * scaling
        assert torch.allclose(Q, ref_q, rtol=0.001, atol=0.001), (
            f"Fused LoRA QKV Q mismatch: max diff {(Q - ref_q).abs().max():.2e}"
        )
        assert torch.allclose(K, ref_k, rtol=0.001, atol=0.001), (
            f"Fused LoRA QKV K mismatch: max diff {(K - ref_k).abs().max():.2e}"
        )
        assert torch.allclose(V, ref_v, rtol=0.001, atol=0.001), (
            f"Fused LoRA QKV V mismatch: max diff {(V - ref_v).abs().max():.2e}"
        )

    def test_fused_lora_qkv_backward(self, device):
        """Fused LoRA QKV should produce correct gradients."""
        from opaque.api.patches.kernels.lora import Opaque_LoRA_QKV

        torch.manual_seed(42)
        batch, seq, hidden, q_out, kv_out, rank = (2, 16, 256, 256, 64, 8)
        scaling = 2.0
        x = torch.randn(batch, seq, hidden, device=device, requires_grad=True)
        Wq = torch.randn(q_out, hidden, device=device)
        Wk = torch.randn(kv_out, hidden, device=device)
        Wv = torch.randn(kv_out, hidden, device=device)
        Aq = torch.randn(hidden, rank, device=device, requires_grad=True)
        Bq = torch.randn(rank, q_out, device=device, requires_grad=True)
        Ak = torch.randn(hidden, rank, device=device, requires_grad=True)
        Bk = torch.randn(rank, kv_out, device=device, requires_grad=True)
        Av = torch.randn(hidden, rank, device=device, requires_grad=True)
        Bv = torch.randn(rank, kv_out, device=device, requires_grad=True)
        Q, K, V = Opaque_LoRA_QKV.apply(
            x, Wq, Aq, Bq, scaling, Wk, Ak, Bk, scaling, Wv, Av, Bv, scaling
        )
        (Q.sum() + K.sum() + V.sum()).backward()
        assert x.grad is not None, "No gradient for input"
        assert not torch.isnan(x.grad).any(), "NaN in input gradients"
        assert Aq.grad is not None, "No gradient for Q LoRA A"
        assert Bq.grad is not None, "No gradient for Q LoRA B"
        assert Ak.grad is not None, "No gradient for K LoRA A"
        assert Bv.grad is not None, "No gradient for V LoRA B"

    @requires_hf_auth
    def test_apply_model_patches_on_peft_model_qkv(self, device):
        """apply_model_patches() should fuse QKV layers on a PEFT-wrapped model."""
        config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        config.num_hidden_layers = 2
        model = AutoModelForCausalLM.from_config(config)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config).to(device)
        apply_model_patches(model, performance=False, compat=True, lora=True)
        layers = model.model.model.layers
        for layer in layers:
            attn = layer.self_attn
            assert hasattr(attn, "_opaque_fused_qkv"), (
                "Attention should have _opaque_fused_qkv after auto-fuse"
            )

    @requires_hf_auth
    def test_fused_lora_qkv_model_forward_backward(self, device):
        """Full model with fused LoRA QKV should produce valid forward+backward."""
        config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        config.num_hidden_layers = 2
        model = AutoModelForCausalLM.from_config(config)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config).to(device)
        apply_model_patches(model, performance=False, compat=True, lora=True)
        input_ids = torch.randint(0, config.vocab_size, (2, 16), device=device)
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        assert not torch.isnan(loss), "NaN loss from fused LoRA QKV model"
        assert loss.item() > 0, "Loss should be positive"
        loss.backward()
        has_grad = False
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                has_grad = True
                assert not torch.isnan(p.grad).any(), f"NaN in gradient for {name}"
        assert has_grad, "No gradients computed"

    def test_qwen2_skips_qkv_fusion(self, device):
        """Qwen2 attention should NOT be fused (has bias=True on Q/K/V)."""
        config = Qwen2Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        model = AutoModelForCausalLM.from_config(config)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config).to(device)
        apply_model_patches(model, performance=False, compat=True, lora=True)
        layers = model.model.model.layers
        for layer in layers:
            attn = layer.self_attn
            assert not hasattr(attn, "_opaque_fused_qkv"), (
                "Qwen2 attention should NOT have fused QKV (bias=True)"
            )

    def test_qwen3_skips_qkv_fusion(self, device):
        """Qwen3 attention should NOT be fused (has q_norm/k_norm)."""
        config = Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        model = AutoModelForCausalLM.from_config(config)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config).to(device)
        apply_model_patches(model, performance=False, compat=True, lora=True)
        layers = model.model.model.layers
        for layer in layers:
            attn = layer.self_attn
            assert not hasattr(attn, "_opaque_fused_qkv"), (
                "Qwen3 attention should NOT have fused QKV (q_norm/k_norm)"
            )

    @requires_hf_auth
    def test_apply_peft_model_patches_manual_qkv(self, device):
        """apply_peft_model_patches() should fuse QKV for manually loaded PEFT models."""
        from opaque.patches.peft import apply_peft_model_patches

        config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        config.num_hidden_layers = 2
        model = AutoModelForCausalLM.from_config(config)
        from peft.mapping_func import get_peft_model as raw_get_peft_model

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        model = raw_get_peft_model(model, lora_config).to(device)
        layers = model.model.model.layers
        assert not hasattr(layers[0].self_attn, "_opaque_fused_qkv"), (
            "QKV should not be fused before apply_peft_model_patches()"
        )
        apply_peft_model_patches(model)
        assert hasattr(layers[0].self_attn, "_opaque_fused_qkv"), (
            "QKV should be fused after apply_peft_model_patches()"
        )

    @requires_hf_auth
    def test_fused_qkv_clipped_grad(self, device):
        """Fused QKV should work end-to-end with clipped_grad (DP-SGD)."""
        config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        config.num_hidden_layers = 2
        model = AutoModelForCausalLM.from_config(config)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config).to(device)
        apply_model_patches(model, performance=False, compat=True, lora=True)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        tokenizer.pad_token = tokenizer.eos_token
        texts = ["Hello world", "Another test", "Third sample", "Last one"]
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
            per_example_loss, argnums=0, batch_argnums=(2, 3, 4), clipping_norm=1.0
        )
        grads, _state = grad_fn(
            trainable, frozen, input_ids, attention_mask, labels, state=clip_state
        )
        assert grads is not None, "No gradients returned"
        assert len(grads.pytree) > 0, "Empty gradient dict"
        for name, g in grads.pytree.items():
            assert not torch.isnan(g).any(), f"NaN in grad for {name}"
