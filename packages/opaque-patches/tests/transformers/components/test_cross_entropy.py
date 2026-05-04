import pytest
from opaque.patches import apply_runtime_patches
import torch
import torch.nn.functional as F

apply_runtime_patches(cross_entropy=True)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Kernel patch compatibility tests require CUDA/Triton",
)

RTOL = 0.0001
ATOL = 0.0001


@pytest.mark.cuda
class TestCrossEntropyPatches:
    """Test that patched cross-entropy loss produces correct outputs."""

    def test_causal_lm_loss_matches_pytorch(self, device):
        """Patched ForCausalLMLoss should match F.cross_entropy reference."""
        from opaque.patches.transformers.components.cross_entropy import (
            _opaque_causal_lm_loss,
        )

        batch, seq_len, vocab_size = (2, 16, 1000)
        logits = torch.randn(batch, seq_len, vocab_size, device=device)
        labels = torch.randint(0, vocab_size, (batch, seq_len), device=device)
        labels[:, -2:] = -100
        loss = _opaque_causal_lm_loss(logits, labels, vocab_size)
        import torch.nn as nn

        labels_ref = nn.functional.pad(labels, (0, 1), value=-100)
        shift_labels = labels_ref[..., 1:].contiguous()
        logits_flat = logits.float().view(-1, vocab_size)
        shift_labels_flat = shift_labels.view(-1)
        ref = F.cross_entropy(logits_flat, shift_labels_flat, ignore_index=-100)
        assert torch.allclose(loss, ref, rtol=0.001, atol=0.001), (
            f"Cross-entropy loss mismatch: got {loss.item():.6f}, expected {ref.item():.6f}"
        )

    def test_causal_lm_loss_with_num_items_in_batch(self, device):
        """Loss with num_items_in_batch should use sum reduction."""
        from opaque.patches.transformers.components.cross_entropy import (
            _opaque_causal_lm_loss,
        )

        batch, seq_len, vocab_size = (2, 16, 1000)
        logits = torch.randn(batch, seq_len, vocab_size, device=device)
        labels = torch.randint(0, vocab_size, (batch, seq_len), device=device)
        import torch.nn as nn

        num_items = torch.tensor(
            batch * (seq_len - 1), dtype=torch.float32, device=device
        )
        loss = _opaque_causal_lm_loss(
            logits, labels, vocab_size, num_items_in_batch=num_items
        )
        labels_ref = nn.functional.pad(labels, (0, 1), value=-100)
        shift_labels = labels_ref[..., 1:].contiguous()
        logits_flat = logits.float().view(-1, vocab_size)
        shift_labels_flat = shift_labels.view(-1)
        ref = (
            F.cross_entropy(
                logits_flat, shift_labels_flat, ignore_index=-100, reduction="sum"
            )
            / num_items
        )
        assert torch.allclose(loss, ref, rtol=0.001, atol=0.001), (
            f"Sum-reduced loss mismatch: got {loss.item():.6f}, expected {ref.item():.6f}"
        )

    def test_backward_through_patched_loss(self, device):
        """Gradients should flow through patched cross-entropy loss."""
        from opaque.patches.transformers.components.cross_entropy import (
            _opaque_causal_lm_loss,
        )

        batch, seq_len, vocab_size = (2, 16, 1000)
        logits = torch.randn(
            batch, seq_len, vocab_size, device=device, requires_grad=True
        )
        labels = torch.randint(0, vocab_size, (batch, seq_len), device=device)
        loss = _opaque_causal_lm_loss(logits, labels, vocab_size)
        loss.backward()
        assert logits.grad is not None, "No gradient through patched loss"
        assert not torch.isnan(logits.grad).any(), "NaN in loss gradients"
        assert not torch.isinf(logits.grad).any(), "Inf in loss gradients"

    def test_loss_mapping_patched(self):
        """LOSS_MAPPING should point to Opaque loss function after patching."""
        from opaque.patches.transformers.components.cross_entropy import (
            _opaque_causal_lm_loss,
        )

        try:
            from transformers.loss.loss_utils import LOSS_MAPPING
        except ImportError:
            pytest.skip("transformers not available")
        if torch.cuda.is_available():
            assert LOSS_MAPPING.get("ForCausalLM") is _opaque_causal_lm_loss
