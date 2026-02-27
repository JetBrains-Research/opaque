"""
RMS LayerNorm Kernel Tests

Tests:
1. Forward pass vs PyTorch reference
2. Backward pass vs PyTorch reference
3. vmap (per-sample grad) vs PyTorch vmap
4. Performance benchmarks (forward-only and forward+backward)
5. Edge cases (zero input, large values)

Config: Mellum-4b scale (hidden_dim=3072)
"""

import pytest
import torch
from torch.func import vmap, grad
from opaque.kernels.rms_layernorm import opaque_rms_norm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

RTOL_FORWARD = 1e-5
ATOL_FORWARD = 1e-5
RTOL_BACKWARD = 1e-4
ATOL_BACKWARD = 1e-5


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def pytorch_rms_layernorm(x, weight, eps=1e-6):
    """PyTorch reference implementation."""
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x * weight


def opaque_rms_layernorm(x, weight, eps=1e-6):
    """Opaque Triton kernel wrapper."""
    return opaque_rms_norm(x, weight, eps=eps)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

class TestRMSLayerNormForward:
    """Test forward pass precision."""

    def test_forward_matches_pytorch(self, assert_precision, mellum_config):
        """Forward: opaque vs pytorch at Mellum scale."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq = mellum_config["seq_len"]
        hidden = mellum_config["hidden_dim"]

        x = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.float32)
        weight = torch.randn(hidden, device="cuda", dtype=torch.float32)

        out_pytorch = pytorch_rms_layernorm(x, weight)
        out_opaque = opaque_rms_layernorm(x, weight)

        print("\nRMSNorm Forward:")
        assert_precision(out_opaque, out_pytorch, rtol=RTOL_FORWARD, atol=ATOL_FORWARD, label="forward")


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------

class TestRMSLayerNormBackward:
    """Test backward pass precision."""

    def test_backward_matches_pytorch(self, assert_precision, mellum_config):
        """Backward: opaque vs pytorch (x.grad and weight.grad)."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq = mellum_config["seq_len"]
        hidden = mellum_config["hidden_dim"]

        # PyTorch reference
        x_pt = torch.randn(
            batch, seq, hidden, device="cuda", dtype=torch.float32, requires_grad=True
        )
        w_pt = torch.randn(
            hidden, device="cuda", dtype=torch.float32, requires_grad=True
        )
        out_pt = pytorch_rms_layernorm(x_pt, w_pt)
        out_pt.sum().backward()

        # Opaque kernel
        x_op = x_pt.detach().clone().requires_grad_(True)
        w_op = w_pt.detach().clone().requires_grad_(True)
        out_op = opaque_rms_layernorm(x_op, w_op)
        out_op.sum().backward()

        print("\nRMSNorm Backward:")
        assert_precision(x_op.grad, x_pt.grad, rtol=RTOL_BACKWARD, atol=ATOL_BACKWARD, label="x.grad")
        assert_precision(w_op.grad, w_pt.grad, rtol=RTOL_BACKWARD, atol=ATOL_BACKWARD, label="weight.grad")


# ---------------------------------------------------------------------------
# Vmap
# ---------------------------------------------------------------------------

class TestRMSLayerNormVmapForward:
    """Test vmap forward: Triton vmap vs PyTorch vmap."""

    def test_vmap_forward_precision(self, assert_precision, mellum_config):
        """Batched forward: opaque Triton vmap vs PyTorch reference."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch = mellum_config["batch_size"]
        seq = mellum_config["seq_len"]
        hidden = mellum_config["hidden_dim"]

        weight = torch.randn(hidden, device="cuda", dtype=torch.float32)
        x = torch.randn(vmap_batch, batch, seq, hidden, device="cuda", dtype=torch.float32)

        out_pt = vmap(lambda xi: pytorch_rms_layernorm(xi, weight))(x)
        out_op = vmap(lambda xi: opaque_rms_layernorm(xi, weight))(x)

        print("\nRMSNorm vmap forward:")
        assert_precision(out_op, out_pt, rtol=RTOL_FORWARD, atol=ATOL_FORWARD, label="vmap_forward")

    def test_vmap_forward_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """Triton vmap forward must be faster or use less memory."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch = mellum_config["batch_size"]
        seq = mellum_config["seq_len"]
        hidden = mellum_config["hidden_dim"]

        weight = torch.randn(hidden, device="cuda", dtype=torch.float32)
        x = torch.randn(vmap_batch, batch, seq, hidden, device="cuda", dtype=torch.float32)

        pt_stats = measure_time_and_memory(lambda xi: vmap(lambda x: pytorch_rms_layernorm(x, weight))(xi), x)
        op_stats = measure_time_and_memory(lambda xi: vmap(lambda x: opaque_rms_layernorm(x, weight))(xi), x)

        assert_perf_benefit(pt_stats, op_stats, label="RMSNorm vmap forward")


class TestRMSLayerNormVmapGrad:
    """Test vmap(grad): per-example gradients — the DP-SGD path."""

    def test_vmap_grad_precision(self, assert_precision, mellum_config):
        """Per-example gradients: opaque Triton vs PyTorch reference."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch = mellum_config["batch_size"]
        seq = mellum_config["seq_len"]
        hidden = mellum_config["hidden_dim"]

        weight = torch.randn(hidden, device="cuda", dtype=torch.float32)
        x = torch.randn(vmap_batch, batch, seq, hidden, device="cuda", dtype=torch.float32)

        def f_pt(xi):
            return pytorch_rms_layernorm(xi, weight).sum()

        def f_op(xi):
            return opaque_rms_layernorm(xi, weight).sum()

        grads_pt = vmap(grad(f_pt))(x)
        grads_op = vmap(grad(f_op))(x)

        print("\nRMSNorm vmap(grad):")
        assert_precision(grads_op, grads_pt, rtol=RTOL_BACKWARD, atol=ATOL_BACKWARD, label="vmap_grad")

    def test_vmap_grad_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """Triton vmap(grad) must be faster or use less memory."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch = mellum_config["batch_size"]
        seq = mellum_config["seq_len"]
        hidden = mellum_config["hidden_dim"]

        weight = torch.randn(hidden, device="cuda", dtype=torch.float32)
        x = torch.randn(vmap_batch, batch, seq, hidden, device="cuda", dtype=torch.float32)

        def make_pt_fn():
            def f(xi):
                return pytorch_rms_layernorm(xi, weight).sum()
            return vmap(grad(f))

        def make_op_fn():
            def f(xi):
                return opaque_rms_layernorm(xi, weight).sum()
            return vmap(grad(f))

        pt_stats = measure_time_and_memory(make_pt_fn(), x)
        op_stats = measure_time_and_memory(make_op_fn(), x)

        assert_perf_benefit(pt_stats, op_stats, label="RMSNorm vmap(grad)")


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

class TestRMSLayerNormPerformance:
    """Benchmark forward-only and forward+backward performance."""

    def test_forward_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """Benchmark forward-only: opaque vs pytorch."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq = mellum_config["seq_len"]
        hidden = mellum_config["hidden_dim"]

        x = torch.randn(
            batch, seq, hidden, device="cuda", dtype=torch.float32,
        )
        weight = torch.randn(hidden, device="cuda", dtype=torch.float32)

        # Forward-only: no requires_grad so measure_time_and_memory skips backward
        pt_stats = measure_time_and_memory(pytorch_rms_layernorm, x, weight)
        op_stats = measure_time_and_memory(opaque_rms_layernorm, x, weight)

        assert_perf_benefit(pt_stats, op_stats, label="RMSNorm forward-only")

    def test_backward_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """Benchmark forward+backward: opaque vs pytorch."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq = mellum_config["seq_len"]
        hidden = mellum_config["hidden_dim"]

        x = torch.randn(
            batch, seq, hidden,
            device="cuda", dtype=torch.float32, requires_grad=True,
        )
        weight = torch.randn(
            hidden, device="cuda", dtype=torch.float32, requires_grad=True,
        )

        pt_stats = measure_time_and_memory(pytorch_rms_layernorm, x, weight)
        op_stats = measure_time_and_memory(opaque_rms_layernorm, x, weight)

        assert_perf_benefit(pt_stats, op_stats, label="RMSNorm forward+backward")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestRMSLayerNormEdgeCases:
    """Test edge cases."""

    def test_zero_input(self):
        """Should handle zero input without NaN/Inf."""
        x = torch.zeros(2, 4, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.ones(128, device="cuda", dtype=torch.bfloat16)

        output = opaque_rms_norm(x, weight, eps=1e-6)
        assert not torch.isnan(output).any(), "NaN in output for zero input"
        assert not torch.isinf(output).any(), "Inf in output for zero input"

    def test_large_values(self):
        """Should handle large values with reasonable precision."""
        x = torch.randn(2, 4, 128, device="cuda", dtype=torch.bfloat16) * 100
        weight = torch.ones(128, device="cuda", dtype=torch.bfloat16)

        output_triton = opaque_rms_norm(x, weight)
        output_pytorch = pytorch_rms_layernorm(x, weight)

        assert torch.allclose(output_triton, output_pytorch, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
