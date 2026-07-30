"""
SwiGLU Kernel Tests

Tests:
1. Forward pass vs PyTorch reference
2. Backward pass vs PyTorch reference
3. vmap forward: Triton vmap vs PyTorch vmap
4. vmap(grad): per-example gradients — the DP-SGD path
5. Forward and backward performance benchmarks

Target precision (bfloat16):
- norm_err: max |a - b| / max(|b|, threshold) < rtol
  where threshold filters out near-zero values that inflate relative errors
- Performance: speedup > 1.0x OR memory reduction > 1.0x for vmap

Config: Mellum-4b scale (intermediate_dim=8256)
"""

import pytest
import torch
import torch.nn.functional as F
from torch.func import grad, vmap

pytest.importorskip("triton")

from opaque.api.patches.kernels.swiglu import Opaque_SwiGLU

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

RTOL_FORWARD = 1e-5
ATOL_FORWARD = 1e-5
RTOL_BACKWARD = 1e-2
ATOL_BACKWARD = 5e-8


def pytorch_swiglu(gate, up):
    """PyTorch reference: SwiGLU = silu(gate) * up."""
    return F.silu(gate) * up


def opaque_swiglu(gate, up):
    """Opaque Triton kernel."""
    return Opaque_SwiGLU.apply(gate, up)


class TestSwiGLUForward:
    """Test forward pass precision."""

    def test_forward_matches_pytorch(self, assert_precision, mellum_config):
        """Forward: opaque vs pytorch."""
        torch.manual_seed(42)
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        gate = torch.randn(batch, seq, dim, device="cuda", dtype=torch.bfloat16)
        up = torch.randn(batch, seq, dim, device="cuda", dtype=torch.bfloat16)

        out_pytorch = pytorch_swiglu(gate, up)
        out_opaque = opaque_swiglu(gate, up)

        print("\nForward precision check:")
        assert_precision(
            out_opaque,
            out_pytorch,
            rtol=RTOL_FORWARD,
            atol=ATOL_FORWARD,
            label="forward",
        )


class TestSwiGLUBackward:
    """Test backward pass precision."""

    def test_backward_matches_pytorch(self, assert_precision, mellum_config):
        """Backward: opaque vs pytorch."""
        torch.manual_seed(42)
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        # PyTorch reference
        gate_pt = torch.randn(
            batch, seq, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        up_pt = torch.randn(
            batch, seq, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        out_pt = pytorch_swiglu(gate_pt, up_pt)
        out_pt.mean().backward()

        # Opaque kernel
        gate_op = gate_pt.detach().clone().requires_grad_(True)
        up_op = up_pt.detach().clone().requires_grad_(True)
        out_op = opaque_swiglu(gate_op, up_op)
        out_op.mean().backward()

        print("\nBackward precision check:")
        assert_precision(
            gate_op.grad,
            gate_pt.grad,
            rtol=RTOL_BACKWARD,
            atol=ATOL_BACKWARD,
            label="gate.grad",
        )
        assert_precision(
            up_op.grad,
            up_pt.grad,
            rtol=RTOL_BACKWARD,
            atol=ATOL_BACKWARD,
            label="up.grad",
        )


class TestSwiGLUVmapForward:
    """Test vmap forward: Triton vmap vs PyTorch vmap."""

    def test_vmap_forward_precision(self, assert_precision, mellum_config):
        """Batched forward: opaque Triton vmap vs PyTorch reference."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        gate = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )
        up = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )

        out_pt = vmap(pytorch_swiglu)(gate, up)
        out_op = vmap(opaque_swiglu)(gate, up)

        print("\nvmap forward precision check:")
        assert_precision(
            out_op, out_pt, rtol=RTOL_FORWARD, atol=ATOL_FORWARD, label="vmap forward"
        )

    def test_vmap_forward_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        """Triton vmap forward must be faster or use less memory than PyTorch."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        gate = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )
        up = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )

        pt_stats = measure_time_and_memory(
            lambda g, u: vmap(pytorch_swiglu)(g, u), gate, up
        )
        op_stats = measure_time_and_memory(
            lambda g, u: vmap(opaque_swiglu)(g, u), gate, up
        )

        assert_perf_benefit(pt_stats, op_stats, label="vmap forward")


class TestSwiGLUVmapGrad:
    """Test vmap(grad): per-example gradients — the DP-SGD path.

    Exercises both Opaque_SwiGLU.vmap() (forward) and
    _SwiGLUBackward.vmap() (backward) with Triton kernels.
    """

    def test_vmap_grad_precision(self, assert_precision, mellum_config):
        """Per-example gradients: opaque Triton vs PyTorch reference."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        gate = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )
        up = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )

        def f_pt(g, u):
            return pytorch_swiglu(g, u).mean()

        def f_op(g, u):
            return opaque_swiglu(g, u).mean()

        grads_pt_gate, grads_pt_up = vmap(grad(f_pt, argnums=(0, 1)))(gate, up)
        grads_op_gate, grads_op_up = vmap(grad(f_op, argnums=(0, 1)))(gate, up)

        print("\nvmap(grad) precision check:")
        assert_precision(
            grads_op_gate,
            grads_pt_gate,
            rtol=RTOL_BACKWARD,
            atol=ATOL_BACKWARD,
            label="vmap(grad) gate",
        )
        assert_precision(
            grads_op_up,
            grads_pt_up,
            rtol=RTOL_BACKWARD,
            atol=ATOL_BACKWARD,
            label="vmap(grad) up",
        )

    def test_vmap_grad_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        """Triton vmap(grad) must be faster or use less memory than PyTorch."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        gate = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )
        up = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )

        def make_pt_fn():
            def f(g, u):
                return pytorch_swiglu(g, u).mean()

            return vmap(grad(f, argnums=(0, 1)))

        def make_op_fn():
            def f(g, u):
                return opaque_swiglu(g, u).mean()

            return vmap(grad(f, argnums=(0, 1)))

        pt_stats = measure_time_and_memory(make_pt_fn(), gate, up)
        op_stats = measure_time_and_memory(make_op_fn(), gate, up)

        assert_perf_benefit(pt_stats, op_stats, label="vmap(grad)")


class TestSwiGLUPerformance:
    """Benchmark forward and backward performance."""

    def test_forward_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        """Forward-only: opaque vs pytorch performance."""
        torch.manual_seed(42)
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        gate = torch.randn(batch, seq, dim, device="cuda", dtype=torch.bfloat16)
        up = torch.randn(batch, seq, dim, device="cuda", dtype=torch.bfloat16)

        pt_stats = measure_time_and_memory(pytorch_swiglu, gate, up)
        op_stats = measure_time_and_memory(opaque_swiglu, gate, up)

        assert_perf_benefit(pt_stats, op_stats, label="forward")

    def test_backward_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        """Forward+backward: opaque vs pytorch performance."""
        torch.manual_seed(42)
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        gate = torch.randn(
            batch, seq, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        up = torch.randn(
            batch, seq, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        pt_stats = measure_time_and_memory(pytorch_swiglu, gate, up)
        op_stats = measure_time_and_memory(opaque_swiglu, gate, up)

        assert_perf_benefit(pt_stats, op_stats, label="backward")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
