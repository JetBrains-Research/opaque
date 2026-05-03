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
class TestGranitePatches:
    """Test kernel patches for Granite models."""

    def test_granite_swiglu_mlp_matches(self, device):
        """Patched GraniteMLP should match PyTorch SwiGLU reference."""
        try:
            from transformers.models.granite.modeling_granite import GraniteMLP
        except ImportError:
            pytest.skip('Granite not available')
        config = AutoConfig.from_pretrained('ibm-granite/granite-3.3-2b-instruct')
        config.num_hidden_layers = 1
        mlp = GraniteMLP(config).to(device)
        x = torch.randn(2, 16, config.hidden_size, device=device, dtype=torch.float32)
        out = mlp(x)
        gate = mlp.gate_proj(x)
        up = mlp.up_proj(x)
        ref = mlp.down_proj(F.silu(gate) * up)
        assert torch.allclose(out, ref, rtol=RTOL, atol=ATOL), f'GraniteMLP output mismatch: max diff {(out - ref).abs().max():.2e}'
