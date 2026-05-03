import pytest
from ...compat._helpers import requires_hf_auth
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
class TestQwen3Patches:
    """Test kernel patches for Qwen3 models."""

    def test_qwen3_swiglu_mlp_matches(self, device):
        """Patched Qwen3MLP should match PyTorch SwiGLU reference."""
        try:
            from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
        except ImportError:
            pytest.skip('Qwen3 not available')
        config = AutoConfig.from_pretrained('Qwen/Qwen3-0.6B')
        config.num_hidden_layers = 1
        mlp = Qwen3MLP(config).to(device)
        x = torch.randn(2, 16, config.hidden_size, device=device, dtype=torch.float32)
        out = mlp(x)
        gate = mlp.gate_proj(x)
        up = mlp.up_proj(x)
        ref = mlp.down_proj(F.silu(gate) * up)
        assert torch.allclose(out, ref, rtol=RTOL, atol=ATOL), f'Qwen3MLP output mismatch: max diff {(out - ref).abs().max():.2e}'
