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
class TestCoherePatches:
    """Test kernel patches for Cohere models (SwiGLU).

    Note: CohereLayerNorm is NOT patched — PyTorch's native F.layer_norm has a
    C++ vmap batching rule that's ~2x faster than our autograd.Function dispatch.
    """

    def test_cohere_swiglu_mlp_matches(self, device):
        """Patched CohereMLP should match PyTorch SwiGLU reference."""
        try:
            from transformers.models.cohere.modeling_cohere import CohereMLP
            from transformers.models.cohere.configuration_cohere import CohereConfig
        except ImportError:
            pytest.skip('Cohere not available')
        config = CohereConfig(hidden_size=256, intermediate_size=512, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2)
        mlp = CohereMLP(config).to(device)
        x = torch.randn(2, 16, config.hidden_size, device=device, dtype=torch.float32)
        out = mlp(x)
        gate = mlp.gate_proj(x)
        up = mlp.up_proj(x)
        ref = mlp.down_proj(F.silu(gate) * up)
        assert torch.allclose(out, ref, rtol=RTOL, atol=ATOL), f'CohereMLP output mismatch: max diff {(out - ref).abs().max():.2e}'
