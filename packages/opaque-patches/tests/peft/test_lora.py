from opaque.patches import apply_runtime_patches
import pytest
import torch
import torch.nn.functional as F

apply_runtime_patches()

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Kernel patch compatibility tests require CUDA/Triton",
)

RTOL = 0.0001
ATOL = 0.0001


@pytest.mark.cuda
class TestLoRAPatches:
    """Test that patched LoRA linear produces correct outputs."""

    def test_lora_forward_matches_peft(self, device):
        """Patched LoRA forward should match PyTorch reference."""
        from peft.tuners.lora import Linear as PeftLoRALinear

        in_features, out_features, rank = (256, 512, 8)
        base_linear = torch.nn.Linear(in_features, out_features, bias=False).to(device)
        lora_layer = PeftLoRALinear(
            base_linear, "default", r=rank, lora_alpha=16, lora_dropout=0.0
        ).to(device)
        x = torch.randn(2, 16, in_features, device=device)
        out = lora_layer(x)
        base_out = base_linear(x)
        A_weight = lora_layer.lora_A["default"].weight
        B_weight = lora_layer.lora_B["default"].weight
        scaling = lora_layer.scaling["default"]
        lora_delta = F.linear(F.linear(x, A_weight), B_weight) * scaling
        ref = base_out + lora_delta
        assert torch.allclose(out, ref, rtol=RTOL, atol=ATOL), (
            f"LoRA forward mismatch: max diff {(out - ref).abs().max():.2e}"
        )

    def test_lora_forward_with_bias(self, device):
        """LoRA forward should correctly handle base layer bias."""
        from peft.tuners.lora import Linear as PeftLoRALinear

        in_features, out_features, rank = (256, 512, 8)
        base_linear = torch.nn.Linear(in_features, out_features, bias=True).to(device)
        lora_layer = PeftLoRALinear(
            base_linear, "default", r=rank, lora_alpha=16, lora_dropout=0.0
        ).to(device)
        x = torch.randn(2, 16, in_features, device=device)
        out = lora_layer(x)
        base_out = base_linear(x)
        A_weight = lora_layer.lora_A["default"].weight
        B_weight = lora_layer.lora_B["default"].weight
        scaling = lora_layer.scaling["default"]
        lora_delta = F.linear(F.linear(x, A_weight), B_weight) * scaling
        ref = base_out + lora_delta
        assert torch.allclose(out, ref, rtol=RTOL, atol=ATOL), (
            f"LoRA with bias mismatch: max diff {(out - ref).abs().max():.2e}"
        )

    def test_backward_through_patched_lora(self, device):
        """Gradients should flow through patched LoRA."""
        from peft.tuners.lora import Linear as PeftLoRALinear

        in_features, out_features, rank = (256, 512, 8)
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
        A_grad = lora_layer.lora_A["default"].weight.grad
        B_grad = lora_layer.lora_B["default"].weight.grad
        assert A_grad is not None, "No gradient for LoRA A"
        assert B_grad is not None, "No gradient for LoRA B"
