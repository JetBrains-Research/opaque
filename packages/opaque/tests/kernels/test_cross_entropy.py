"""
Cross Entropy Kernel Tests

Tests:
1. Forward pass vs PyTorch reference
2. Backward pass vs PyTorch reference
3. vmap (per-sample grad) vs PyTorch vmap
4. Forward and backward performance benchmarks

Config: Mellum-4b scale (uses mellum_config from conftest).
Mellum vocab=98304 (>65536) exercises the chunked cross-entropy path.
"""

import pytest
import torch
import torch.nn.functional as F
from torch.func import vmap, grad

from opaque.kernels.cross_entropy import Opaque_CrossEntropy

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

# Cross entropy tolerances (accumulation over large vocab dimension)
RTOL_CE_FORWARD = 1e-4
RTOL_CE_BACKWARD = 1e-4


# ============================================================================
# Reference Implementations
# ============================================================================

def pytorch_cross_entropy(logits, labels):
    """PyTorch reference: F.cross_entropy with reduction='mean'."""
    vocab_size = logits.shape[-1]
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)
    loss = F.cross_entropy(logits_flat, labels_flat, reduction="mean")
    return loss


def opaque_cross_entropy(logits, labels):
    """Opaque kernel: Opaque_CrossEntropy with masked mean for vmap compatibility."""
    losses, _ = Opaque_CrossEntropy.apply(logits, labels)
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

    def test_forward_matches_pytorch(self, precision_error, mellum_config):
        """Forward: opaque vs pytorch at mellum scale (chunked path)."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        vocab = mellum_config["vocab_size"]

        logits = torch.randn(batch, seq_len, vocab, device="cuda", dtype=torch.float32)
        labels = torch.randint(0, vocab, (batch, seq_len), device="cuda")

        out_pytorch = pytorch_cross_entropy(logits, labels)
        out_opaque = opaque_cross_entropy(logits, labels)

        err = precision_error(out_opaque.unsqueeze(0), out_pytorch.unsqueeze(0), threshold=1e-4)
        print(f"\nCE Forward (vocab={vocab}): abs={err['abs_err']:.2e}, rel={err['rel_err']:.2e} (target: <{RTOL_CE_FORWARD:.0e})")

        assert err["rel_err"] < RTOL_CE_FORWARD, f"Forward rel_err {err['rel_err']:.2e} >= {RTOL_CE_FORWARD:.0e}"

    @pytest.mark.parametrize("vocab_size", [128256])
    def test_forward_other_vocab_sizes(self, precision_error, vocab_size):
        """Forward with other large vocab sizes (e.g. LLaMA 3 128K)."""
        torch.manual_seed(42)

        logits = torch.randn(2, 64, vocab_size, device="cuda", dtype=torch.float32)
        labels = torch.randint(0, vocab_size, (2, 64), device="cuda")

        losses_op, _ = Opaque_CrossEntropy.apply(logits, labels)

        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)
        losses_pt = F.cross_entropy(logits_flat, labels_flat, reduction="none")

        err = precision_error(losses_op.reshape(-1), losses_pt, threshold=1e-4)
        print(f"\nCE Forward (vocab={vocab_size}): abs={err['abs_err']:.2e}, rel={err['rel_err']:.2e}")

        assert err["rel_err"] < 1e-4, f"Forward rel_err {err['rel_err']:.2e} >= 1e-4"


# ============================================================================
# Backward Pass Tests
# ============================================================================

class TestCrossEntropyBackward:
    """Test backward pass precision."""

    def test_backward_matches_pytorch(self, precision_error, mellum_config):
        """Backward: opaque vs pytorch logits.grad at mellum scale."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        vocab = mellum_config["vocab_size"]

        # PyTorch reference
        logits_pt = torch.randn(
            batch, seq_len, vocab, device="cuda", dtype=torch.float32, requires_grad=True
        )
        labels = torch.randint(0, vocab, (batch, seq_len), device="cuda")

        out_pt = pytorch_cross_entropy(logits_pt, labels)
        out_pt.backward()

        # Opaque kernel
        logits_op = logits_pt.detach().clone().requires_grad_(True)
        out_op = opaque_cross_entropy(logits_op, labels)
        out_op.backward()

        err = precision_error(logits_op.grad, logits_pt.grad, threshold=1e-4)

        print(f"\nCE Backward (vocab={vocab}):")
        print(f"  logits.grad: abs={err['abs_err']:.2e}, rel={err['rel_err']:.2e} (target: <{RTOL_CE_BACKWARD:.0e})")

        assert err["rel_err"] < RTOL_CE_BACKWARD, f"logits.grad rel_err {err['rel_err']:.2e} >= {RTOL_CE_BACKWARD:.0e}"

    def test_backward_ignores_masked_labels(self, mellum_config):
        """Verify -100 labels produce zero gradient (not softmax probs)."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        vocab = mellum_config["vocab_size"]

        logits = torch.randn(batch, seq_len, vocab, device="cuda", dtype=torch.float32, requires_grad=True)
        labels = torch.randint(0, vocab, (batch, seq_len), device="cuda")
        labels[:, -10:] = -100  # Mask last 10 positions

        losses, _ = Opaque_CrossEntropy.apply(logits, labels)
        losses.sum().backward()

        # Gradients at -100 positions must be exactly zero
        masked_grad = logits.grad[:, -10:, :]
        assert masked_grad.abs().max() == 0.0, (
            f"Non-zero grad at -100 positions: max={masked_grad.abs().max():.2e}"
        )


# ============================================================================
# Vmap Tests
# ============================================================================

class TestCrossEntropyVmapForward:
    """Test vmap forward: Triton vmap vs PyTorch vmap."""

    def test_vmap_forward_precision(self, precision_error, mellum_config):
        """Batched forward: opaque Triton vmap vs PyTorch reference."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        vocab = mellum_config["vocab_size"]
        vmap_batch = mellum_config["vmap_batch"]

        logits = torch.randn(
            vmap_batch, batch, seq_len, vocab,
            device="cuda", dtype=torch.float32,
        )
        labels = torch.randint(0, vocab, (vmap_batch, batch, seq_len), device="cuda")

        out_pt = vmap(pytorch_cross_entropy, in_dims=(0, 0))(logits, labels)
        out_op = vmap(opaque_cross_entropy, in_dims=(0, 0))(logits, labels)

        err = precision_error(out_op.unsqueeze(0), out_pt.unsqueeze(0), threshold=1e-4)
        print(f"\nCE vmap forward (vocab={vocab}): abs={err['abs_err']:.2e}, rel={err['rel_err']:.2e} (target: <{RTOL_CE_FORWARD:.0e})")

        assert err["rel_err"] < RTOL_CE_FORWARD, f"vmap forward rel_err {err['rel_err']:.2e} >= {RTOL_CE_FORWARD:.0e}"

    def test_vmap_forward_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """Triton vmap forward must be faster or use less memory."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        vocab = mellum_config["vocab_size"]
        vmap_batch = mellum_config["vmap_batch"]

        logits = torch.randn(
            vmap_batch, batch, seq_len, vocab,
            device="cuda", dtype=torch.float32,
        )
        labels = torch.randint(0, vocab, (vmap_batch, batch, seq_len), device="cuda")

        pt_stats = measure_time_and_memory(
            lambda l, t: vmap(pytorch_cross_entropy, in_dims=(0, 0))(l, t), logits, labels
        )
        op_stats = measure_time_and_memory(
            lambda l, t: vmap(opaque_cross_entropy, in_dims=(0, 0))(l, t), logits, labels
        )

        assert_perf_benefit(pt_stats, op_stats, label="CE vmap forward")


class TestCrossEntropyVmapGrad:
    """Test vmap(grad): per-example gradients — the DP-SGD path."""

    def test_vmap_grad_precision(self, precision_error, mellum_config):
        """Per-example gradients: opaque Triton vs PyTorch reference."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        vocab = mellum_config["vocab_size"]
        vmap_batch = mellum_config["vmap_batch"]

        logits = torch.randn(
            vmap_batch, batch, seq_len, vocab,
            device="cuda", dtype=torch.float32,
        )
        labels = torch.randint(0, vocab, (vmap_batch, batch, seq_len), device="cuda")

        def f_pt(l, t):
            return pytorch_cross_entropy(l, t)

        def f_op(l, t):
            return opaque_cross_entropy(l, t)

        # grad w.r.t. logits only (argnums=0), labels are not differentiable
        grads_pt = vmap(grad(f_pt, argnums=0), in_dims=(0, 0))(logits, labels)
        grads_op = vmap(grad(f_op, argnums=0), in_dims=(0, 0))(logits, labels)

        err = precision_error(grads_op, grads_pt, threshold=1e-4)
        print(f"\nCE vmap(grad) (vocab={vocab}): abs={err['abs_err']:.2e}, rel={err['rel_err']:.2e} (target: <{RTOL_CE_BACKWARD:.0e})")

        assert err["rel_err"] < RTOL_CE_BACKWARD, f"vmap(grad) rel_err {err['rel_err']:.2e} >= {RTOL_CE_BACKWARD:.0e}"

    def test_vmap_grad_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """Triton vmap(grad) must be faster or use less memory."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        vocab = mellum_config["vocab_size"]
        vmap_batch = mellum_config["vmap_batch"]

        logits = torch.randn(
            vmap_batch, batch, seq_len, vocab,
            device="cuda", dtype=torch.float32,
        )
        labels = torch.randint(0, vocab, (vmap_batch, batch, seq_len), device="cuda")

        def make_pt_fn():
            def f(l, t):
                return pytorch_cross_entropy(l, t)
            return vmap(grad(f, argnums=0), in_dims=(0, 0))

        def make_op_fn():
            def f(l, t):
                return opaque_cross_entropy(l, t)
            return vmap(grad(f, argnums=0), in_dims=(0, 0))

        pt_stats = measure_time_and_memory(make_pt_fn(), logits, labels)
        op_stats = measure_time_and_memory(make_op_fn(), logits, labels)

        assert_perf_benefit(pt_stats, op_stats, label="CE vmap(grad)")


# ============================================================================
# Performance Tests
# ============================================================================

class TestCrossEntropyPerformance:
    """Benchmark forward and backward performance."""

    def test_forward_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """Forward-only: opaque vs pytorch performance."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        vocab = mellum_config["vocab_size"]

        logits = torch.randn(batch, seq_len, vocab, device="cuda", dtype=torch.float32)
        labels = torch.randint(0, vocab, (batch, seq_len), device="cuda")

        def pytorch_fn(l, t):
            return pytorch_cross_entropy(l, t)

        def opaque_fn(l, t):
            return opaque_cross_entropy(l, t)

        pt_stats = measure_time_and_memory(pytorch_fn, logits, labels)
        op_stats = measure_time_and_memory(opaque_fn, logits, labels)

        assert_perf_benefit(pt_stats, op_stats, label="CE forward", max_perf_overhead=0.60)

    def test_backward_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """Forward+backward: opaque vs pytorch performance."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq_len = mellum_config["seq_len"]
        vocab = mellum_config["vocab_size"]

        logits = torch.randn(
            batch, seq_len, vocab, device="cuda", dtype=torch.float32, requires_grad=True
        )
        labels = torch.randint(0, vocab, (batch, seq_len), device="cuda")

        def pytorch_fn(l, t):
            return pytorch_cross_entropy(l, t)

        def opaque_fn(l, t):
            return opaque_cross_entropy(l, t)

        pt_stats = measure_time_and_memory(pytorch_fn, logits, labels)
        op_stats = measure_time_and_memory(opaque_fn, logits, labels)

        assert_perf_benefit(pt_stats, op_stats, label="CE backward")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
