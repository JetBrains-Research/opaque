import pytest
import torch
import torch.nn.functional as F
from opaque_test_support import requires_hf_auth
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2Config,
    Qwen3Config,
)

from opaque.api.engine.clipping import clipped_grad
from opaque.functional import make_functional
from opaque.patches import apply_model_patches, apply_runtime_patches

apply_runtime_patches()

RTOL = 0.0001
ATOL = 0.0001


def test_qwen3_qkv_fusion_routes_and_falls_back_on_cpu():
    """Qwen3 receives the dedicated fusion wrapper and keeps its CPU fallback."""
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    model = get_peft_model(
        AutoModelForCausalLM.from_config(config),
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj"],
        ),
    )
    apply_model_patches(model, performance=False, compat=True, lora=True)

    attn = model.model.model.layers[0].self_attn
    assert hasattr(attn, "_opaque_fused_qkv")
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    assert torch.isfinite(model(input_ids).logits).all()


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
        bq = torch.randn(q_out, device=device)
        bk = torch.randn(kv_out, device=device)
        bv = torch.randn(kv_out, device=device)
        Aq = torch.randn(hidden, rank, device=device)
        Bq = torch.randn(rank, q_out, device=device)
        Ak = torch.randn(hidden, rank, device=device)
        Bk = torch.randn(rank, kv_out, device=device)
        Av = torch.randn(hidden, rank, device=device)
        Bv = torch.randn(rank, kv_out, device=device)
        Q, K, V = Opaque_LoRA_QKV.apply(
            x,
            Wq,
            Aq,
            Bq,
            scaling,
            bq,
            Wk,
            Ak,
            Bk,
            scaling,
            bk,
            Wv,
            Av,
            Bv,
            scaling,
            bv,
        )
        ref_q = F.linear(x, Wq, bq) + x @ Aq @ Bq * scaling
        ref_k = F.linear(x, Wk, bk) + x @ Ak @ Bk * scaling
        ref_v = F.linear(x, Wv, bv) + x @ Av @ Bv * scaling
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
        bq = torch.randn(q_out, device=device)
        bk = None
        bv = torch.randn(kv_out, device=device)
        Aq = torch.randn(hidden, rank, device=device, requires_grad=True)
        Bq = torch.randn(rank, q_out, device=device, requires_grad=True)
        Ak = torch.randn(hidden, rank, device=device, requires_grad=True)
        Bk = torch.randn(rank, kv_out, device=device, requires_grad=True)
        Av = torch.randn(hidden, rank, device=device, requires_grad=True)
        Bv = torch.randn(rank, kv_out, device=device, requires_grad=True)
        Q, K, V = Opaque_LoRA_QKV.apply(
            x,
            Wq,
            Aq,
            Bq,
            scaling,
            bq,
            Wk,
            Ak,
            Bk,
            scaling,
            bk,
            Wv,
            Av,
            Bv,
            scaling,
            bv,
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

    def test_qwen2_fuses_biased_qkv_projections(self, device):
        """Qwen2 attention fuses its frozen biased Q/K/V projections."""
        config = Qwen2Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            attention_bias=True,
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
            assert hasattr(attn, "_opaque_fused_qkv")
            assert attn.q_proj.base_layer.bias is not None
            assert not attn.q_proj.base_layer.bias.requires_grad

    def test_qwen2_skips_qkv_fusion_with_trainable_bias(self, device):
        """QKV fusion preserves gradients for PEFT configurations that train biases."""
        config = Qwen2Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            attention_bias=True,
        )
        model = get_peft_model(
            AutoModelForCausalLM.from_config(config),
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                bias="all",
                target_modules=["q_proj", "k_proj", "v_proj"],
            ),
        ).to(device)
        apply_model_patches(model, performance=False, compat=True, lora=True)
        attn = model.model.model.layers[0].self_attn
        assert attn.q_proj.base_layer.bias.requires_grad
        assert not hasattr(attn, "_opaque_fused_qkv")

    def test_qwen2_biased_qkv_clipped_grad(self, device):
        """Fused biased Qwen2 QKV projections support per-example gradients."""
        config = Qwen2Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            attention_bias=True,
        )
        model = get_peft_model(
            AutoModelForCausalLM.from_config(config),
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj"],
            ),
        ).to(device)
        apply_model_patches(model, performance=False, compat=True, lora=True)

        input_ids = torch.randint(0, config.vocab_size, (2, 8), device=device)
        outputs = model(input_ids, labels=input_ids)
        assert torch.isfinite(outputs.loss)
        outputs.loss.backward()

        fmodel, trainable, frozen = make_functional(
            model, disable_autograd_tracking=True, partition_trainable=True
        )

        def per_example_loss(trainable_params, frozen_params, ids, labels):
            return fmodel(
                {**frozen_params, **trainable_params}, ids, labels=labels
            ).loss

        grad_fn, clip_state = clipped_grad(
            per_example_loss, argnums=0, batch_argnums=(2, 3), clipping_norm=1.0
        )
        grads, _ = grad_fn(trainable, frozen, input_ids, input_ids, state=clip_state)
        assert all(torch.isfinite(grad).all() for grad in grads.pytree.values())

    def test_qwen3_fuses_qkv_with_head_norms(self, device):
        """Qwen3 QKV fusion preserves its post-projection head normalization."""
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
            assert hasattr(attn, "_opaque_fused_qkv")

    def test_qwen3_fused_qkv_matches_unfused_logits_and_gradients(self, device):
        """Fused Qwen3 projections match the unfused Q/K-normalized attention."""
        torch.manual_seed(0)
        config = Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
        )
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        unfused = get_peft_model(
            AutoModelForCausalLM.from_config(config), lora_config
        ).to(device)
        fused = get_peft_model(
            AutoModelForCausalLM.from_config(config), lora_config
        ).to(device)
        fused.load_state_dict(unfused.state_dict())
        for module in unfused.modules():
            if hasattr(module, "lora_B"):
                module.lora_B["default"].weight.data.normal_(std=0.02)
        fused.load_state_dict(unfused.state_dict())
        apply_model_patches(fused, performance=False, compat=True, lora=True)

        input_ids = torch.randint(0, config.vocab_size, (2, 8), device=device)
        unfused_logits = unfused(input_ids).logits
        fused_logits = fused(input_ids).logits
        torch.testing.assert_close(fused_logits, unfused_logits, rtol=RTOL, atol=ATOL)

        unfused_cache = unfused(input_ids, use_cache=True).past_key_values
        fused_cache = fused(input_ids, use_cache=True).past_key_values
        next_input_ids = torch.randint(0, config.vocab_size, (2, 1), device=device)
        torch.testing.assert_close(
            fused(next_input_ids, past_key_values=fused_cache).logits,
            unfused(next_input_ids, past_key_values=unfused_cache).logits,
            rtol=RTOL,
            atol=ATOL,
        )

        unfused_logits.square().mean().backward()
        fused_logits.square().mean().backward()
        unfused_grads = dict(unfused.named_parameters())
        for name, parameter in fused.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None
                torch.testing.assert_close(
                    parameter.grad, unfused_grads[name].grad, rtol=RTOL, atol=ATOL
                )

    def test_qwen3_fused_qkv_clipped_grad_matches_unfused(self, device):
        """Fused Qwen3 QKV supports the same per-example gradients as unfused."""
        torch.manual_seed(0)
        config = Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
        )
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        unfused = get_peft_model(
            AutoModelForCausalLM.from_config(config), lora_config
        ).to(device)
        fused = get_peft_model(
            AutoModelForCausalLM.from_config(config), lora_config
        ).to(device)
        fused.load_state_dict(unfused.state_dict())
        apply_model_patches(fused, performance=False, compat=True, lora=True)
        input_ids = torch.randint(0, config.vocab_size, (2, 8), device=device)

        def per_example_grads(model):
            fmodel, trainable, frozen = make_functional(
                model, disable_autograd_tracking=True, partition_trainable=True
            )

            def loss(trainable_params, frozen_params, ids, labels):
                return fmodel(
                    {**frozen_params, **trainable_params}, ids, labels=labels
                ).loss

            grad_fn, clip_state = clipped_grad(
                loss, argnums=0, batch_argnums=(2, 3), clipping_norm=1.0
            )
            return grad_fn(trainable, frozen, input_ids, input_ids, state=clip_state)[
                0
            ].pytree

        fused_grads = per_example_grads(fused)
        unfused_grads = per_example_grads(unfused)
        for name, gradient in fused_grads.items():
            assert torch.isfinite(gradient).all()
            torch.testing.assert_close(
                gradient, unfused_grads[name], rtol=RTOL, atol=ATOL
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
