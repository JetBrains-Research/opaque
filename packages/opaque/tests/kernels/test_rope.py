"""
RoPE (Rotary Position Embedding) Kernel Tests

Tests:
1. Forward pass vs PyTorch reference
2. Backward pass vs PyTorch reference
3. vmap (per-sample grad) vs PyTorch vmap
4. Forward and backward performance benchmarks

Config: Mellum-like dims (batch=2, seq=128, n_heads=24, head_dim=128)
"""

import pytest
import torch

from opaque.kernels.rope_embedding import NewStyleRoPEEmbedding

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

# Mellum-like dims for RoPE tests
BATCH = 2
SEQ_LEN = 128

RTOL_ROPE = 5e-4


# ============================================================================
# Helpers
# ============================================================================

def rotate_half(x):
    """Standard rotate_half for RoPE."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def generate_cos_sin(seq_len, head_dim, device="cuda", dtype=torch.float32):
    """Generate cos/sin caches for RoPE.

    Returns:
        cos: (seq_len, head_dim // 2)
        sin: (seq_len, head_dim // 2)
    """
    freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device)
    freqs = torch.outer(positions, freqs)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def pytorch_rope(Q, cos, sin):
    """PyTorch reference RoPE implementation.

    Args:
        Q: (batch, seq_len, n_heads, head_dim)
        cos: (seq_len, head_dim // 2)
        sin: (seq_len, head_dim // 2)
    """
    batch, seq_len, n_heads, head_dim = Q.shape

    cos_expanded = cos[None, :, None, :].expand(batch, seq_len, n_heads, -1)
    sin_expanded = sin[None, :, None, :].expand(batch, seq_len, n_heads, -1)

    cos_full = torch.cat([cos_expanded, cos_expanded], dim=-1)
    sin_full = torch.cat([sin_expanded, sin_expanded], dim=-1)

    return Q * cos_full + rotate_half(Q) * sin_full


def opaque_rope(Q, cos, sin):
    """Opaque Triton kernel (NewStyleRoPEEmbedding)."""
    return NewStyleRoPEEmbedding.apply(Q, cos, sin)


# ============================================================================
# Forward Pass Tests
# ============================================================================

class TestRoPEForward:
    """Test forward pass precision."""

    def test_forward_matches_pytorch(self, precision_error, mellum_config):
        """Forward: opaque vs pytorch."""
        torch.manual_seed(42)
        N_HEADS = mellum_config["n_heads"]
        HEAD_DIM = mellum_config["head_dim"]

        Q = torch.randn(BATCH, SEQ_LEN, N_HEADS, HEAD_DIM, device="cuda", dtype=torch.float32)
        cos, sin = generate_cos_sin(SEQ_LEN, HEAD_DIM)

        out_opaque = opaque_rope(Q, cos, sin)
        out_pytorch = pytorch_rope(Q, cos, sin)

        err = precision_error(out_opaque, out_pytorch, threshold=1e-4)
        print(f"\nRoPE Forward: abs={err['abs_err']:.2e}, rel={err['rel_err']:.2e} (target: <{RTOL_ROPE:.0e})")

        assert err["rel_err"] < RTOL_ROPE, f"Forward rel_err {err['rel_err']:.2e} >= {RTOL_ROPE:.0e}"


# ============================================================================
# Backward Pass Tests
# ============================================================================

class TestRoPEBackward:
    """Test backward pass precision."""

    def test_backward_matches_pytorch(self, precision_error, mellum_config):
        """Backward: opaque vs pytorch Q.grad."""
        torch.manual_seed(42)
        N_HEADS = mellum_config["n_heads"]
        HEAD_DIM = mellum_config["head_dim"]

        # PyTorch reference
        Q_pt = torch.randn(BATCH, SEQ_LEN, N_HEADS, HEAD_DIM, device="cuda", dtype=torch.float32, requires_grad=True)
        cos, sin = generate_cos_sin(SEQ_LEN, HEAD_DIM)
        out_pt = pytorch_rope(Q_pt, cos, sin)
        out_pt.sum().backward()

        # Opaque kernel
        Q_op = Q_pt.detach().clone().requires_grad_(True)
        out_op = opaque_rope(Q_op, cos, sin)
        out_op.sum().backward()

        err = precision_error(Q_op.grad, Q_pt.grad, threshold=1e-4)
        print(f"\nRoPE Backward Q.grad: abs={err['abs_err']:.2e}, rel={err['rel_err']:.2e} (target: <{RTOL_ROPE:.0e})")

        assert err["rel_err"] < RTOL_ROPE, f"Q.grad rel_err {err['rel_err']:.2e} >= {RTOL_ROPE:.0e}"


# ============================================================================
# Vmap Tests
# ============================================================================

class TestRoPEVmap:
    """Test vmap (per-sample gradient) precision and performance."""

    def test_vmap_precision(self, precision_error, mellum_config):
        """vmap: opaque vs pytorch vmap forward + backward."""
        torch.manual_seed(42)
        N_HEADS = mellum_config["n_heads"]
        HEAD_DIM = mellum_config["head_dim"]
        VMAP_BATCH = mellum_config["vmap_batch"]

        cos, sin = generate_cos_sin(SEQ_LEN, HEAD_DIM)

        # PyTorch vmap
        Q_pt = torch.randn(VMAP_BATCH, BATCH, SEQ_LEN, N_HEADS, HEAD_DIM, device="cuda", dtype=torch.float32, requires_grad=True)
        out_pt = torch.vmap(lambda q: pytorch_rope(q, cos, sin))(Q_pt)
        out_pt.sum().backward()

        # Opaque vmap
        Q_op = Q_pt.detach().clone().requires_grad_(True)
        out_op = torch.vmap(lambda q: opaque_rope(q, cos, sin))(Q_op)
        out_op.sum().backward()

        fwd_err = precision_error(out_op, out_pt, threshold=1e-4)
        bwd_err = precision_error(Q_op.grad, Q_pt.grad, threshold=1e-4)

        print(f"\nRoPE vmap:")
        print(f"  forward: abs={fwd_err['abs_err']:.2e}, rel={fwd_err['rel_err']:.2e} (target: <{RTOL_ROPE:.0e})")
        print(f"  Q.grad:  abs={bwd_err['abs_err']:.2e}, rel={bwd_err['rel_err']:.2e} (target: <{RTOL_ROPE:.0e})")

        assert fwd_err["rel_err"] < RTOL_ROPE, f"vmap forward rel_err {fwd_err['rel_err']:.2e} >= {RTOL_ROPE:.0e}"
        assert bwd_err["rel_err"] < RTOL_ROPE, f"vmap Q.grad rel_err {bwd_err['rel_err']:.2e} >= {RTOL_ROPE:.0e}"

    def test_vmap_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """vmap: opaque should be faster or use less memory than pytorch vmap."""
        torch.manual_seed(42)
        N_HEADS = mellum_config["n_heads"]
        HEAD_DIM = mellum_config["head_dim"]
        VMAP_BATCH = mellum_config["vmap_batch"]

        cos, sin = generate_cos_sin(SEQ_LEN, HEAD_DIM)

        Q = torch.randn(VMAP_BATCH, BATCH, SEQ_LEN, N_HEADS, HEAD_DIM, device="cuda", dtype=torch.float32, requires_grad=True)

        def pytorch_fn(q):
            return torch.vmap(lambda qi: pytorch_rope(qi, cos, sin))(q)

        def opaque_fn(q):
            return torch.vmap(lambda qi: opaque_rope(qi, cos, sin))(q)

        pt_stats = measure_time_and_memory(pytorch_fn, Q)
        op_stats = measure_time_and_memory(opaque_fn, Q)

        assert_perf_benefit(pt_stats, op_stats, label="RoPE vmap")


# ============================================================================
# Performance Tests
# ============================================================================

class TestRoPEPerformance:
    """Benchmark forward and backward performance."""

    def test_forward_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """Forward-only: opaque vs pytorch performance."""
        torch.manual_seed(42)
        N_HEADS = mellum_config["n_heads"]
        HEAD_DIM = mellum_config["head_dim"]

        cos, sin = generate_cos_sin(SEQ_LEN, HEAD_DIM)
        Q = torch.randn(BATCH, SEQ_LEN, N_HEADS, HEAD_DIM, device="cuda", dtype=torch.float32)

        def pytorch_fn(q):
            return pytorch_rope(q, cos, sin)

        def opaque_fn(q):
            return opaque_rope(q, cos, sin)

        pt_stats = measure_time_and_memory(pytorch_fn, Q)
        op_stats = measure_time_and_memory(opaque_fn, Q)

        assert_perf_benefit(pt_stats, op_stats, label="RoPE forward")

    def test_backward_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """Forward+backward: opaque vs pytorch performance."""
        torch.manual_seed(42)
        N_HEADS = mellum_config["n_heads"]
        HEAD_DIM = mellum_config["head_dim"]

        cos, sin = generate_cos_sin(SEQ_LEN, HEAD_DIM)
        Q = torch.randn(BATCH, SEQ_LEN, N_HEADS, HEAD_DIM, device="cuda", dtype=torch.float32, requires_grad=True)

        def pytorch_fn(q):
            return pytorch_rope(q, cos, sin)

        def opaque_fn(q):
            return opaque_rope(q, cos, sin)

        pt_stats = measure_time_and_memory(pytorch_fn, Q)
        op_stats = measure_time_and_memory(opaque_fn, Q)

        assert_perf_benefit(pt_stats, op_stats, label="RoPE backward")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
