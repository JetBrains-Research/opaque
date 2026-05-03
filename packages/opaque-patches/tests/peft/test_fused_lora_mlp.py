from opaque.patches import apply_model_patches, apply_runtime_patches
import pytest
from tests._helpers import requires_hf_auth

apply_runtime_patches()
import torch
import torch.nn.functional as F
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Kernel patch compatibility tests require CUDA/Triton')
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from opaque.clipping import clipped_grad
from opaque.functional import make_functional
RTOL = 0.0001
ATOL = 0.0001


@pytest.mark.cuda
class TestFusedLoRAMLP:
    """Test fused LoRA MLP patching via Opaque_LoRA_MLP kernel."""

    def test_fused_lora_mlp_forward(self, device):
        """Fused LoRA MLP forward should match PyTorch matmul reference."""
        from opaque.patches.kernels.lora import Opaque_LoRA_MLP
        torch.manual_seed(42)
        batch, seq, hidden, intermediate, rank = (2, 16, 256, 512, 8)
        scaling = 2.0
        x = torch.randn(batch, seq, hidden, device=device, dtype=torch.float32)
        Wg = torch.randn(intermediate, hidden, device=device)
        Wu = torch.randn(intermediate, hidden, device=device)
        Wd = torch.randn(hidden, intermediate, device=device)
        Ag = torch.randn(hidden, rank, device=device)
        Bg = torch.randn(rank, intermediate, device=device)
        Au = torch.randn(hidden, rank, device=device)
        Bu = torch.randn(rank, intermediate, device=device)
        Ad = torch.randn(intermediate, rank, device=device)
        Bd = torch.randn(rank, hidden, device=device)
        out_fused, _, _, _ = Opaque_LoRA_MLP.apply(x, Wg, Ag, Bg, scaling, Wu, Au, Bu, scaling, Wd, Ad, Bd, scaling, 0)
        gate = F.linear(x, Wg) + x @ Ag @ Bg * scaling
        up = F.linear(x, Wu) + x @ Au @ Bu * scaling
        h = F.silu(gate) * up
        ref = F.linear(h, Wd) + h @ Ad @ Bd * scaling
        assert torch.allclose(out_fused, ref, rtol=0.001, atol=0.001), f'Fused LoRA MLP output mismatch: max diff {(out_fused - ref).abs().max():.2e}'

    def test_fused_lora_mlp_backward(self, device):
        """Fused LoRA MLP should produce correct gradients."""
        from opaque.patches.kernels.lora import Opaque_LoRA_MLP
        torch.manual_seed(42)
        batch, seq, hidden, intermediate, rank = (2, 16, 256, 512, 8)
        scaling = 2.0
        x = torch.randn(batch, seq, hidden, device=device, requires_grad=True)
        Wg = torch.randn(intermediate, hidden, device=device)
        Wu = torch.randn(intermediate, hidden, device=device)
        Wd = torch.randn(hidden, intermediate, device=device)
        Ag = torch.randn(hidden, rank, device=device, requires_grad=True)
        Bg = torch.randn(rank, intermediate, device=device, requires_grad=True)
        Au = torch.randn(hidden, rank, device=device, requires_grad=True)
        Bu = torch.randn(rank, intermediate, device=device, requires_grad=True)
        Ad = torch.randn(intermediate, rank, device=device, requires_grad=True)
        Bd = torch.randn(rank, hidden, device=device, requires_grad=True)
        out, _, _, _ = Opaque_LoRA_MLP.apply(x, Wg, Ag, Bg, scaling, Wu, Au, Bu, scaling, Wd, Ad, Bd, scaling, 0)
        out.sum().backward()
        assert x.grad is not None, 'No gradient for input'
        assert not torch.isnan(x.grad).any(), 'NaN in input gradients'
        assert Ag.grad is not None, 'No gradient for gate LoRA A'
        assert Bd.grad is not None, 'No gradient for down LoRA B'

    @requires_hf_auth
    def test_auto_fuse_on_get_peft_model(self, device):
        """get_peft_model should auto-fuse MLP layers with LoRA on gate/up/down."""
        config = AutoConfig.from_pretrained('meta-llama/Llama-3.2-1B')
        config.num_hidden_layers = 2
        model = AutoModelForCausalLM.from_config(config)
        lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=['gate_proj', 'up_proj', 'down_proj'])
        apply_model_patches(model, performance=False, compat=True)
        model = get_peft_model(model, lora_config).to(device)
        layers = model.model.model.layers
        for layer in layers:
            mlp = layer.mlp
            assert 'forward' in vars(mlp), 'MLP forward should be fused'

    @requires_hf_auth
    def test_fused_lora_mlp_model_forward_backward(self, device):
        """Full model with fused LoRA MLP should produce valid forward+backward."""
        config = AutoConfig.from_pretrained('meta-llama/Llama-3.2-1B')
        config.num_hidden_layers = 2
        model = AutoModelForCausalLM.from_config(config)
        lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=['gate_proj', 'up_proj', 'down_proj'])
        apply_model_patches(model, performance=False, compat=True)
        model = get_peft_model(model, lora_config).to(device)
        input_ids = torch.randint(0, config.vocab_size, (2, 16), device=device)
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        assert not torch.isnan(loss), 'NaN loss from fused LoRA MLP model'
        assert loss.item() > 0, 'Loss should be positive'
        loss.backward()
        has_grad = False
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                has_grad = True
                assert not torch.isnan(p.grad).any(), f'NaN in gradient for {name}'
        assert has_grad, 'No gradients computed'

    @requires_hf_auth
    def test_patch_lora_model_manual(self, device):
        """patch_lora_model() should work for manually loaded PEFT models."""
        from opaque.patches.peft import patch_lora_model
        config = AutoConfig.from_pretrained('meta-llama/Llama-3.2-1B')
        config.num_hidden_layers = 2
        model = AutoModelForCausalLM.from_config(config)
        from peft.mapping_func import get_peft_model as raw_get_peft_model
        lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=['gate_proj', 'up_proj', 'down_proj'])
        model = raw_get_peft_model(model, lora_config).to(device)
        layers = model.model.model.layers
        assert 'forward' not in vars(layers[0].mlp), 'MLP should not be fused before patch_lora_model()'
        patch_lora_model(model)
        assert 'forward' in vars(layers[0].mlp), 'MLP should be fused after patch_lora_model()'
