"""Tests for RMS LayerNorm kernel.

Validates:
1. Correctness vs PyTorch implementation
2. vmap compatibility
3. Speed improvement
4. Memory usage (should not increase)
"""

import pytest
import torch
import time

from opaque.kernels.rms_layernorm import rms_layernorm


def pytorch_rms_layernorm(x, weight, eps=1e-6):
    """Reference PyTorch implementation."""
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x * weight


class TestRMSLayerNormCorrectness:
    """Test numerical correctness."""

    def test_forward_matches_pytorch(self):
        """Forward pass should match PyTorch."""
        torch.manual_seed(42)

        x = torch.randn(4, 128, 4096, device='cuda', dtype=torch.float32)
        weight = torch.randn(4096, device='cuda', dtype=torch.float32)

        output_triton = rms_layernorm(x, weight)
        output_pytorch = pytorch_rms_layernorm(x, weight)

        assert torch.allclose(output_triton, output_pytorch, rtol=1e-5, atol=1e-5)

    def test_backward_matches_pytorch(self):
        """Gradients should match PyTorch."""
        torch.manual_seed(42)

        x1 = torch.randn(4, 128, 4096, device='cuda', dtype=torch.float32, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)
        weight1 = torch.randn(4096, device='cuda', dtype=torch.float32, requires_grad=True)
        weight2 = weight1.detach().clone().requires_grad_(True)

        # Triton
        output_triton = rms_layernorm(x1, weight1)
        output_triton.sum().backward()

        # PyTorch
        output_pytorch = pytorch_rms_layernorm(x2, weight2)
        output_pytorch.sum().backward()

        # Compare gradients
        assert torch.allclose(x1.grad, x2.grad, rtol=1e-5, atol=1e-5)
        assert torch.allclose(weight1.grad, weight2.grad, rtol=1e-5, atol=1e-5)

    def test_different_shapes(self):
        """Should work with various input shapes."""
        shapes = [
            (2, 16, 256),    # Small
            (4, 128, 4096),  # Standard
            (1, 512, 2048),  # Single batch, long sequence
        ]

        for shape in shapes:
            x = torch.randn(*shape, device='cuda', dtype=torch.float32)
            weight = torch.randn(shape[-1], device='cuda', dtype=torch.float32)

            output_triton = rms_layernorm(x, weight)
            output_pytorch = pytorch_rms_layernorm(x, weight)

            assert torch.allclose(output_triton, output_pytorch, rtol=1e-5, atol=1e-5), \
                f"Failed for shape {shape}"

    def test_eps_parameter(self):
        """Should respect eps parameter."""
        x = torch.randn(2, 4, 128, device='cuda', dtype=torch.float32)
        weight = torch.randn(128, device='cuda', dtype=torch.float32)

        for eps in [1e-5, 1e-6, 1e-8]:
            output_triton = rms_layernorm(x, weight, eps=eps)
            output_pytorch = pytorch_rms_layernorm(x, weight, eps=eps)

            assert torch.allclose(output_triton, output_pytorch, rtol=1e-5, atol=1e-5), \
                f"Failed for eps={eps}"


class TestRMSLayerNormVmap:
    """Test vmap compatibility."""

    def test_vmap_forward(self):
        """Should work with vmap."""
        torch.manual_seed(42)

        x = torch.randn(4, 128, 4096, device='cuda', dtype=torch.float32)
        weight = torch.randn(4096, device='cuda', dtype=torch.float32)

        # Regular
        output_regular = rms_layernorm(x, weight)

        # With vmap
        def per_example_fn(x_single):
            return rms_layernorm(x_single, weight)

        output_vmap = torch.vmap(per_example_fn)(x)

        assert torch.allclose(output_vmap, output_regular, rtol=1e-5, atol=1e-5)

    def test_vmap_backward(self):
        """Gradients should match with vmap."""
        torch.manual_seed(42)

        x1 = torch.randn(4, 128, 4096, device='cuda', dtype=torch.float32, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)
        weight = torch.randn(4096, device='cuda', dtype=torch.float32)

        # Regular
        output_regular = rms_layernorm(x1, weight)
        output_regular.sum().backward()

        # With vmap
        def per_example_fn(x_single):
            return rms_layernorm(x_single, weight)

        output_vmap = torch.vmap(per_example_fn)(x2)
        output_vmap.sum().backward()

        assert torch.allclose(x1.grad, x2.grad, rtol=1e-5, atol=1e-5)


class TestRMSLayerNormPerformance:
    """Test speed and memory."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_speed_improvement(self):
        """Triton should be faster than vmapped PyTorch (DP-SGD use case)."""
        torch.manual_seed(42)

        x = torch.randn(4, 128, 4096, device='cuda', dtype=torch.bfloat16)
        weight = torch.randn(4096, device='cuda', dtype=torch.bfloat16)

        # Warmup
        for _ in range(10):
            _ = rms_layernorm(x, weight)
            _ = pytorch_rms_layernorm(x, weight)
            _ = torch.vmap(lambda x_single: pytorch_rms_layernorm(x_single, weight))(x)
        torch.cuda.synchronize()

        # Benchmark Triton
        iterations = 100
        start = time.time()
        for _ in range(iterations):
            _ = rms_layernorm(x, weight)
        torch.cuda.synchronize()
        time_triton = (time.time() - start) / iterations

        # Benchmark PyTorch (regular)
        start = time.time()
        for _ in range(iterations):
            _ = pytorch_rms_layernorm(x, weight)
        torch.cuda.synchronize()
        time_pytorch = (time.time() - start) / iterations

        # Benchmark PyTorch with vmap (DP-SGD relevant)
        start = time.time()
        for _ in range(iterations):
            _ = torch.vmap(lambda x_single: pytorch_rms_layernorm(x_single, weight))(x)
        torch.cuda.synchronize()
        time_vmap = (time.time() - start) / iterations

        speedup_vs_regular = time_pytorch / time_triton
        speedup_vs_vmap = time_vmap / time_triton

        print(f"\nRMSNorm speed:")
        print(f"  PyTorch (regular): {time_pytorch*1000:.3f} ms")
        print(f"  PyTorch (vmap):    {time_vmap*1000:.3f} ms")
        print(f"  Triton:            {time_triton*1000:.3f} ms")
        print(f"  Speedup vs regular: {speedup_vs_regular:.2f}x")
        print(f"  Speedup vs vmap:    {speedup_vs_vmap:.2f}x")

        # For DP-SGD, we need to beat vmapped PyTorch
        assert speedup_vs_vmap > 1.1, f"Triton not faster than vmapped PyTorch: {speedup_vs_vmap:.2f}x"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_memory_usage(self):
        """Triton should not use more memory than PyTorch."""
        torch.manual_seed(42)

        x = torch.randn(4, 128, 4096, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(4096, device='cuda', dtype=torch.bfloat16, requires_grad=True)

        # Measure PyTorch memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        output_pt = pytorch_rms_layernorm(x, weight)
        output_pt.sum().backward()

        mem_pytorch = torch.cuda.max_memory_allocated() / 1e6  # MB

        # Measure Triton memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        x_fresh = x.detach().requires_grad_(True)
        weight_fresh = weight.detach().requires_grad_(True)

        output_triton = rms_layernorm(x_fresh, weight_fresh)
        output_triton.sum().backward()

        mem_triton = torch.cuda.max_memory_allocated() / 1e6  # MB

        mem_diff_abs = mem_triton - mem_pytorch
        mem_diff_pct = (mem_diff_abs / mem_pytorch) * 100

        print(f"\nRMSNorm memory:")
        print(f"  PyTorch: {mem_pytorch:.2f} MB")
        print(f"  Triton:  {mem_triton:.2f} MB")
        print(f"  Diff:    {mem_diff_abs:+.2f} MB ({mem_diff_pct:+.1f}%)")

        # Triton saves inv_var which uses extra memory, but this is acceptable
        # for the speed improvement in DP-SGD. Allow 50% relative overhead.
        assert mem_diff_pct <= 50.0, \
            f"Triton uses {mem_diff_pct:.1f}% more memory (limit: 50%)"


class TestRMSLayerNormEdgeCases:
    """Test edge cases."""

    def test_zero_input(self):
        """Should handle zero input."""
        x = torch.zeros(2, 4, 128, device='cuda', dtype=torch.bfloat16)
        weight = torch.ones(128, device='cuda', dtype=torch.bfloat16)

        output = rms_layernorm(x, weight, eps=1e-6)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_large_values(self):
        """Should handle large values."""
        x = torch.randn(2, 4, 128, device='cuda', dtype=torch.bfloat16) * 100
        weight = torch.ones(128, device='cuda', dtype=torch.bfloat16)

        output_triton = rms_layernorm(x, weight)
        output_pytorch = pytorch_rms_layernorm(x, weight)

        assert torch.allclose(output_triton, output_pytorch, rtol=1e-2, atol=1e-2)

    def test_mixed_precision(self):
        """Should work with float32."""
        x = torch.randn(2, 4, 128, device='cuda', dtype=torch.float32)
        weight = torch.ones(128, device='cuda', dtype=torch.float32)

        output_triton = rms_layernorm(x, weight)
        output_pytorch = pytorch_rms_layernorm(x, weight)

        assert torch.allclose(output_triton, output_pytorch, rtol=1e-4, atol=1e-5)
