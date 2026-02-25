"""
Cross Entropy Kernel Tests

Tests:
1. Forward pass vs PyTorch reference
2. Backward pass vs PyTorch reference
3. vmap (per-sample grad) vs PyTorch vmap
4. Forward and backward performance benchmarks

Config: Mellum-relevant scale (vocab=32000, batch=4, seq=128)
Cross entropy has accumulation over large vocab, so rtol=1e-4 is used.
"""

import pytest
import torch
import torch.nn.functional as F

from opaque.kernels.cross_entropy import NewStyleCrossEntropy

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

# Cross entropy tolerances (accumulation over large vocab dimension)
RTOL_CE_FORWARD = 1e-4
RTOL_CE_BACKWARD = 1e-4

# Test dimensions (vocab=32000 to keep memory manageable)
CE_BATCH = 4
CE_SEQ = 128
CE_VOCAB = 32000


# ============================================================================
# Reference Implementations
# ============================================================================

def pytorch_cross_entropy(logits, labels):
    """PyTorch reference: F.cross_entropy with reduction='mean'."""
    batch_seq = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)
    loss = F.cross_entropy(logits_flat, labels_flat, reduction="mean")
    return loss


def opaque_cross_entropy(logits, labels):
    """Opaque kernel: NewStyleCrossEntropy with masked mean for vmap compatibility."""
    losses, _ = NewStyleCrossEntropy.apply(logits, labels)
    # For vmap compatibility, avoid data-dependent control flow
    mask = (labels != -100).float()
    n_valid = mask.sum()
    masked_losses = losses * mask
    return masked_losses.sum() / torch.clamp(n_valid, min=1.0)


# ============================================================================
# Forward Pass Tests
# ============================================================================

class TestCrossEntropyForward:
    """Test forward pass precision."""

    def test_forward_matches_pytorch(self, precision_error):
        """Forward: opaque vs pytorch."""
        torch.manual_seed(42)

        logits = torch.randn(CE_BATCH, CE_SEQ, CE_VOCAB, device="cuda", dtype=torch.float32)
        labels = torch.randint(0, CE_VOCAB, (CE_BATCH, CE_SEQ), device="cuda")

        out_pytorch = pytorch_cross_entropy(logits, labels)
        out_opaque = opaque_cross_entropy(logits, labels)

        err = precision_error(out_opaque.unsqueeze(0), out_pytorch.unsqueeze(0), threshold=1e-4)
        print(f"\nCE Forward: abs={err['abs_err']:.2e}, rel={err['rel_err']:.2e} (target: <{RTOL_CE_FORWARD:.0e})")

        assert err["rel_err"] < RTOL_CE_FORWARD, f"Forward rel_err {err['rel_err']:.2e} >= {RTOL_CE_FORWARD:.0e}"


# ============================================================================
# Backward Pass Tests
# ============================================================================

class TestCrossEntropyBackward:
    """Test backward pass precision."""

    def test_backward_matches_pytorch(self, precision_error):
        """Backward: opaque vs pytorch logits.grad."""
        torch.manual_seed(42)

        # PyTorch reference
        logits_pt = torch.randn(
            CE_BATCH, CE_SEQ, CE_VOCAB, device="cuda", dtype=torch.float32, requires_grad=True
        )
        labels = torch.randint(0, CE_VOCAB, (CE_BATCH, CE_SEQ), device="cuda")

        out_pt = pytorch_cross_entropy(logits_pt, labels)
        out_pt.backward()

        # Opaque kernel
        logits_op = logits_pt.detach().clone().requires_grad_(True)
        out_op = opaque_cross_entropy(logits_op, labels)
        out_op.backward()

        err = precision_error(logits_op.grad, logits_pt.grad, threshold=1e-4)

        print(f"\nCE Backward:")
        print(f"  logits.grad: abs={err['abs_err']:.2e}, rel={err['rel_err']:.2e} (target: <{RTOL_CE_BACKWARD:.0e})")

        assert err["rel_err"] < RTOL_CE_BACKWARD, f"logits.grad rel_err {err['rel_err']:.2e} >= {RTOL_CE_BACKWARD:.0e}"


# ============================================================================
# Vmap Tests
# ============================================================================

class TestCrossEntropyVmap:
    """Test vmap (per-sample gradient) precision and performance."""

    def test_vmap_precision(self, precision_error, mellum_config):
        """vmap: opaque vs pytorch vmap."""
        torch.manual_seed(42)

        vmap_batch = mellum_config["vmap_batch"]
        logits = torch.randn(
            vmap_batch, CE_BATCH, CE_SEQ, CE_VOCAB,
            device="cuda", dtype=torch.float32, requires_grad=True,
        )
        labels = torch.randint(0, CE_VOCAB, (vmap_batch, CE_BATCH, CE_SEQ), device="cuda")

        # PyTorch vmap
        logits_pt = logits.detach().clone().requires_grad_(True)
        out_pt = torch.vmap(pytorch_cross_entropy, in_dims=(0, 0))(logits_pt, labels)
        out_pt.sum().backward()

        # Opaque vmap
        logits_op = logits.detach().clone().requires_grad_(True)
        out_op = torch.vmap(opaque_cross_entropy, in_dims=(0, 0))(logits_op, labels)
        out_op.sum().backward()

        fwd_err = precision_error(out_op, out_pt, threshold=1e-4)
        grad_err = precision_error(logits_op.grad, logits_pt.grad, threshold=1e-4)

        print(f"\nCE vmap:")
        print(f"  forward:     abs={fwd_err['abs_err']:.2e}, rel={fwd_err['rel_err']:.2e} (target: <{RTOL_CE_FORWARD:.0e})")
        print(f"  logits.grad: abs={grad_err['abs_err']:.2e}, rel={grad_err['rel_err']:.2e} (target: <{RTOL_CE_BACKWARD:.0e})")

        assert fwd_err["rel_err"] < RTOL_CE_FORWARD, f"vmap forward rel_err {fwd_err['rel_err']:.2e} >= {RTOL_CE_FORWARD:.0e}"
        assert grad_err["rel_err"] < RTOL_CE_BACKWARD, f"vmap logits.grad rel_err {grad_err['rel_err']:.2e} >= {RTOL_CE_BACKWARD:.0e}"

    def test_vmap_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """vmap: opaque should be faster or use less memory than pytorch vmap."""
        torch.manual_seed(42)

        vmap_batch = mellum_config["vmap_batch"]
        logits = torch.randn(
            vmap_batch, CE_BATCH, CE_SEQ, CE_VOCAB,
            device="cuda", dtype=torch.float32, requires_grad=True,
        )
        labels = torch.randint(0, CE_VOCAB, (vmap_batch, CE_BATCH, CE_SEQ), device="cuda")

        def pytorch_fn(l, t):
            return torch.vmap(pytorch_cross_entropy, in_dims=(0, 0))(l, t)

        def opaque_fn(l, t):
            return torch.vmap(opaque_cross_entropy, in_dims=(0, 0))(l, t)

        pt_stats = measure_time_and_memory(pytorch_fn, logits, labels)
        op_stats = measure_time_and_memory(opaque_fn, logits, labels)

        assert_perf_benefit(pt_stats, op_stats, label="CE vmap")


# ============================================================================
# Performance Tests
# ============================================================================

class TestCrossEntropyPerformance:
    """Benchmark forward and backward performance."""

    def test_forward_performance(self, measure_time_and_memory, assert_perf_benefit):
        """Forward-only: opaque vs pytorch performance."""
        torch.manual_seed(42)

        logits = torch.randn(CE_BATCH, CE_SEQ, CE_VOCAB, device="cuda", dtype=torch.float32)
        labels = torch.randint(0, CE_VOCAB, (CE_BATCH, CE_SEQ), device="cuda")

        def pytorch_fn(l, t):
            return pytorch_cross_entropy(l, t)

        def opaque_fn(l, t):
            return opaque_cross_entropy(l, t)

        pt_stats = measure_time_and_memory(pytorch_fn, logits, labels)
        op_stats = measure_time_and_memory(opaque_fn, logits, labels)

        assert_perf_benefit(pt_stats, op_stats, label="CE forward", max_perf_overhead=0.60)

    def test_backward_performance(self, measure_time_and_memory, assert_perf_benefit):
        """Forward+backward: opaque vs pytorch performance."""
        torch.manual_seed(42)

        logits = torch.randn(
            CE_BATCH, CE_SEQ, CE_VOCAB, device="cuda", dtype=torch.float32, requires_grad=True
        )
        labels = torch.randint(0, CE_VOCAB, (CE_BATCH, CE_SEQ), device="cuda")

        def pytorch_fn(l, t):
            return pytorch_cross_entropy(l, t)

        def opaque_fn(l, t):
            return opaque_cross_entropy(l, t)

        pt_stats = measure_time_and_memory(pytorch_fn, logits, labels)
        op_stats = measure_time_and_memory(opaque_fn, logits, labels)

        assert_perf_benefit(pt_stats, op_stats, label="CE backward")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
