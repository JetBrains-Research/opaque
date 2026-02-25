"""
SwiGLU Kernel Tests

Tests:
1. Forward pass vs PyTorch reference
2. Backward pass vs PyTorch reference
3. vmap (per-sample grad) vs PyTorch vmap
4. Forward and backward performance benchmarks

Target precision (float32):
- norm_err: max |a - b| / max(|b|, threshold) < rtol
  where threshold filters out near-zero values that inflate relative errors
- Performance: speedup > 1.0x OR memory reduction > 1.0x for vmap

Config: Mellum-4b scale (intermediate_dim=8256)
"""

import pytest
import torch
import torch.nn.functional as F

from opaque.kernels.swiglu import NewStyleSwiGLU

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

RTOL_FORWARD = 1e-5
RTOL_BACKWARD = 1e-3


def pytorch_swiglu(gate, up):
    """PyTorch reference: SwiGLU = silu(gate) * up."""
    return F.silu(gate) * up


def opaque_swiglu(gate, up):
    """Opaque Triton kernel."""
    return NewStyleSwiGLU.apply(gate, up)


class TestSwiGLUForward:
    """Test forward pass precision."""

    def test_forward_matches_pytorch(self, precision_error, mellum_config):
        """Forward: opaque vs pytorch."""
        torch.manual_seed(42)
        batch, seq, dim = mellum_config["batch_size"], mellum_config["seq_len"], mellum_config["intermediate_dim"]

        gate = torch.randn(batch, seq, dim, device="cuda", dtype=torch.float32)
        up = torch.randn(batch, seq, dim, device="cuda", dtype=torch.float32)

        out_pytorch = pytorch_swiglu(gate, up)
        out_opaque = opaque_swiglu(gate, up)

        err = precision_error(out_opaque, out_pytorch, threshold=1e-4)
        print(f"\nForward: abs={err['abs_err']:.2e}, rel={err['rel_err']:.2e} (target: <{RTOL_FORWARD:.0e})")

        assert err["rel_err"] < RTOL_FORWARD, f"Forward rel_err {err['rel_err']:.2e} >= {RTOL_FORWARD:.0e}"


class TestSwiGLUBackward:
    """Test backward pass precision."""

    def test_backward_matches_pytorch(self, precision_error, mellum_config):
        """Backward: opaque vs pytorch."""
        torch.manual_seed(42)
        batch, seq, dim = mellum_config["batch_size"], mellum_config["seq_len"], mellum_config["intermediate_dim"]

        # PyTorch reference
        gate_pt = torch.randn(batch, seq, dim, device="cuda", dtype=torch.float32, requires_grad=True)
        up_pt = torch.randn(batch, seq, dim, device="cuda", dtype=torch.float32, requires_grad=True)
        out_pt = pytorch_swiglu(gate_pt, up_pt)
        out_pt.sum().backward()

        # Opaque kernel
        gate_op = gate_pt.detach().clone().requires_grad_(True)
        up_op = up_pt.detach().clone().requires_grad_(True)
        out_op = opaque_swiglu(gate_op, up_op)
        out_op.sum().backward()

        gate_err = precision_error(gate_op.grad, gate_pt.grad, threshold=1e-4)
        up_err = precision_error(up_op.grad, up_pt.grad, threshold=1e-4)

        print(f"\nBackward:")
        print(f"  gate.grad: abs={gate_err['abs_err']:.2e}, rel={gate_err['rel_err']:.2e}, {gate_err['pct_large']:.1f}% > thresh (target: rel<{RTOL_BACKWARD:.0e})")
        print(f"  up.grad:   abs={up_err['abs_err']:.2e}, rel={up_err['rel_err']:.2e}, {up_err['pct_large']:.1f}% > thresh (target: rel<{RTOL_BACKWARD:.0e})")

        assert gate_err["rel_err"] < RTOL_BACKWARD, f"gate.grad rel_err {gate_err['rel_err']:.2e} >= {RTOL_BACKWARD:.0e}"
        assert up_err["rel_err"] < RTOL_BACKWARD, f"up.grad rel_err {up_err['rel_err']:.2e} >= {RTOL_BACKWARD:.0e}"


class TestSwiGLUVmap:
    """Test vmap (per-sample gradient) precision and performance."""

    def test_vmap_precision(self, precision_error, mellum_config):
        """vmap: opaque vs pytorch vmap."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch, seq, dim = mellum_config["batch_size"], mellum_config["seq_len"], mellum_config["intermediate_dim"]

        gate = torch.randn(vmap_batch, batch, seq, dim, device="cuda", dtype=torch.float32, requires_grad=True)
        up = torch.randn(vmap_batch, batch, seq, dim, device="cuda", dtype=torch.float32, requires_grad=True)

        # PyTorch vmap
        gate_pt = gate.detach().clone().requires_grad_(True)
        up_pt = up.detach().clone().requires_grad_(True)
        out_pt = torch.vmap(pytorch_swiglu)(gate_pt, up_pt)
        out_pt.sum().backward()

        # Opaque vmap
        gate_op = gate.detach().clone().requires_grad_(True)
        up_op = up.detach().clone().requires_grad_(True)
        out_op = torch.vmap(opaque_swiglu)(gate_op, up_op)
        out_op.sum().backward()

        fwd_err = precision_error(out_op, out_pt, threshold=1e-4)
        gate_err = precision_error(gate_op.grad, gate_pt.grad, threshold=1e-4)
        up_err = precision_error(up_op.grad, up_pt.grad, threshold=1e-4)

        print(f"\nvmap precision:")
        print(f"  forward:   abs={fwd_err['abs_err']:.2e}, rel={fwd_err['rel_err']:.2e} (target: <{RTOL_FORWARD:.0e})")
        print(f"  gate.grad: abs={gate_err['abs_err']:.2e}, rel={gate_err['rel_err']:.2e} (target: <{RTOL_BACKWARD:.0e})")
        print(f"  up.grad:   abs={up_err['abs_err']:.2e}, rel={up_err['rel_err']:.2e} (target: <{RTOL_BACKWARD:.0e})")

        assert fwd_err["rel_err"] < RTOL_FORWARD, f"vmap forward rel_err {fwd_err['rel_err']:.2e} >= {RTOL_FORWARD:.0e}"
        assert gate_err["rel_err"] < RTOL_BACKWARD, f"vmap gate.grad rel_err {gate_err['rel_err']:.2e} >= {RTOL_BACKWARD:.0e}"
        assert up_err["rel_err"] < RTOL_BACKWARD, f"vmap up.grad rel_err {up_err['rel_err']:.2e} >= {RTOL_BACKWARD:.0e}"

    def test_vmap_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """vmap: opaque should be faster or use less memory than pytorch vmap."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch, seq, dim = mellum_config["batch_size"], mellum_config["seq_len"], mellum_config["intermediate_dim"]

        gate = torch.randn(vmap_batch, batch, seq, dim, device="cuda", dtype=torch.float32, requires_grad=True)
        up = torch.randn(vmap_batch, batch, seq, dim, device="cuda", dtype=torch.float32, requires_grad=True)

        def pytorch_vmap_fn(g, u):
            return torch.vmap(pytorch_swiglu)(g, u)

        def opaque_vmap_fn(g, u):
            return torch.vmap(opaque_swiglu)(g, u)

        pt_stats = measure_time_and_memory(pytorch_vmap_fn, gate, up)
        op_stats = measure_time_and_memory(opaque_vmap_fn, gate, up)

        assert_perf_benefit(pt_stats, op_stats, label="vmap")


class TestSwiGLUPerformance:
    """Benchmark forward and backward performance."""

    def test_forward_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """Forward-only: opaque vs pytorch performance."""
        torch.manual_seed(42)
        batch, seq, dim = mellum_config["batch_size"], mellum_config["seq_len"], mellum_config["intermediate_dim"]

        gate = torch.randn(batch, seq, dim, device="cuda", dtype=torch.float32)
        up = torch.randn(batch, seq, dim, device="cuda", dtype=torch.float32)

        pt_stats = measure_time_and_memory(pytorch_swiglu, gate, up)
        op_stats = measure_time_and_memory(opaque_swiglu, gate, up)

        assert_perf_benefit(pt_stats, op_stats, label="forward")

    def test_backward_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """Forward+backward: opaque vs pytorch performance."""
        torch.manual_seed(42)
        batch, seq, dim = mellum_config["batch_size"], mellum_config["seq_len"], mellum_config["intermediate_dim"]

        gate = torch.randn(batch, seq, dim, device="cuda", dtype=torch.float32, requires_grad=True)
        up = torch.randn(batch, seq, dim, device="cuda", dtype=torch.float32, requires_grad=True)

        pt_stats = measure_time_and_memory(pytorch_swiglu, gate, up)
        op_stats = measure_time_and_memory(opaque_swiglu, gate, up)

        assert_perf_benefit(pt_stats, op_stats, label="backward")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
