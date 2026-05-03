import pytest
from ...compat._helpers import requires_hf_auth
from opaque.patches import apply_runtime_patches

apply_runtime_patches(use_fused_loss=True)
import torch
import torch.nn.functional as F
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Kernel patch compatibility tests require CUDA/Triton')
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from opaque.clipping import clipped_grad
from opaque.functional import make_functional
RTOL = 0.0001
ATOL = 0.0001


class TestCPUFallback:
    """Test that patched kernels fall back to original on CPU."""

    @requires_hf_auth
    def test_swiglu_mlp_cpu(self):
        """Patched LlamaMLP should produce correct output on CPU."""
        from transformers.models.llama.modeling_llama import LlamaMLP
        config = AutoConfig.from_pretrained('meta-llama/Llama-3.2-1B')
        config.num_hidden_layers = 1
        mlp = LlamaMLP(config)
        x = torch.randn(2, 16, config.hidden_size)
        out = mlp(x)
        gate = mlp.gate_proj(x)
        up = mlp.up_proj(x)
        ref = mlp.down_proj(F.silu(gate) * up)
        assert torch.allclose(out, ref, rtol=RTOL, atol=ATOL), f'CPU SwiGLU mismatch: max diff {(out - ref).abs().max():.2e}'

    @requires_hf_auth
    def test_geglu_exact_mlp_cpu(self):
        """Patched GemmaMLP should produce correct output on CPU."""
        try:
            from transformers.models.gemma.modeling_gemma import GemmaMLP
        except ImportError:
            pytest.skip('Gemma not available')
        config = AutoConfig.from_pretrained('google/gemma-2b')
        config.num_hidden_layers = 1
        mlp = GemmaMLP(config)
        x = torch.randn(2, 16, config.hidden_size)
        out = mlp(x)
        gate = mlp.gate_proj(x)
        up = mlp.up_proj(x)
        ref = mlp.down_proj(F.gelu(gate, approximate='none') * up)
        assert torch.allclose(out, ref, rtol=RTOL, atol=ATOL), f'CPU GeGLU exact mismatch: max diff {(out - ref).abs().max():.2e}'

    def test_cross_entropy_loss_cpu(self):
        """Patched CE loss should produce correct output on CPU."""
        from opaque.patches.transformers.components import _opaque_causal_lm_loss
        batch, seq_len, vocab_size = (2, 16, 1000)
        logits = torch.randn(batch, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch, seq_len))
        labels[:, -2:] = -100
        loss = _opaque_causal_lm_loss(logits, labels, vocab_size)
        import torch.nn as nn
        labels_ref = nn.functional.pad(labels, (0, 1), value=-100)
        shift_labels = labels_ref[..., 1:].contiguous()
        logits_flat = logits.float().view(-1, vocab_size)
        shift_labels_flat = shift_labels.view(-1)
        ref = F.cross_entropy(logits_flat, shift_labels_flat, ignore_index=-100)
        assert torch.allclose(loss, ref, rtol=0.001, atol=0.001), f'CPU CE loss mismatch: got {loss.item():.6f}, expected {ref.item():.6f}'

    def test_lora_linear_cpu(self):
        """Patched LoRA linear should produce correct output on CPU."""
        from peft.tuners.lora import Linear as PeftLoRALinear
        in_features, out_features, rank = (256, 512, 8)
        base_linear = torch.nn.Linear(in_features, out_features, bias=False)
        lora_layer = PeftLoRALinear(base_linear, 'default', r=rank, lora_alpha=16, lora_dropout=0.0)
        x = torch.randn(2, 16, in_features)
        out = lora_layer(x)
        base_out = base_linear(x)
        A_weight = lora_layer.lora_A['default'].weight
        B_weight = lora_layer.lora_B['default'].weight
        scaling = lora_layer.scaling['default']
        lora_delta = F.linear(F.linear(x, A_weight), B_weight) * scaling
        ref = base_out + lora_delta
        assert torch.allclose(out, ref, rtol=RTOL, atol=ATOL), f'CPU LoRA mismatch: max diff {(out - ref).abs().max():.2e}'

    @requires_hf_auth
    def test_full_model_cpu_forward_backward(self):
        """Full Llama model with LoRA should forward+backward correctly on CPU."""
        config = AutoConfig.from_pretrained('meta-llama/Llama-3.2-1B')
        config.num_hidden_layers = 1
        model = AutoModelForCausalLM.from_config(config)
        lora_config = LoraConfig(r=4, lora_alpha=8, target_modules=['q_proj', 'v_proj'], lora_dropout=0.0)
        model = get_peft_model(model, lora_config)
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        outputs = model(input_ids, labels=input_ids)
        assert not torch.isnan(outputs.loss), 'NaN loss on CPU'
        outputs.loss.backward()
        has_grad = any((p.grad is not None for p in model.parameters() if p.requires_grad))
        assert has_grad, 'No gradients computed on CPU'
