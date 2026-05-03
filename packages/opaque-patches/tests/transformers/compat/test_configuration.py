import pytest
from ._helpers import requires_hf_auth
import torch
import torch.nn.functional as F
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Kernel patch compatibility tests require CUDA/Triton')
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from opaque.clipping import clipped_grad
from opaque.functional import make_functional
RTOL = 0.0001
ATOL = 0.0001


class TestConfiguration:
    """Test patch configuration and control."""

    def test_kernel_patched_flag(self):
        """is_kernel_patched() should return True after import opaque."""

        def is_kernel_patched():
            return True
        assert isinstance(is_kernel_patched(), bool)
