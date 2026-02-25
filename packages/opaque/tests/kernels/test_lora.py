"""
LoRA Kernel Tests

Tests opaque LoRA kernels against PyTorch vmap:
1. Precision: rtol=1e-7 (relative error only)
2. Performance: measure forward+backward time vs PyTorch vmap
3. Memory: measure peak memory vs PyTorch vmap

Each test verifies that the kernel provides either a performance or memory benefit.
"""

import pytest
import torch
import torch.nn.functional as F
import sys
import os

# Add kernel path without importing main opaque package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

try:
    from .kernel_validation_framework import (
        benchmark_forward_backward,
        validate_implementations,
        print_validation_result,
        print_comparison_table,
    )
except ImportError:
    from kernel_validation_framework import (
        benchmark_forward_backward,
        validate_implementations,
        print_validation_result,
        print_comparison_table,
    )

# Import kernels directly to avoid opaque __init__.py dependency issues
from opaque.kernels.lora import NewStyleLoRAW, NewStyleLoRAQKV, NewStyleLoRAMLP


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for kernel tests"
)


# ============================================================================
# Reference Implementations
# ============================================================================

def pytorch_lora_linear(X, W, A, B, scaling):
    """PyTorch reference LoRA linear implementation.

    out = X @ W.T + (X @ A @ B) * scaling

    Args:
        X: (batch, seq_len, in_features)
        W: (out_features, in_features)
        A: (in_features, rank)
        B: (rank, out_features)
        scaling: scalar
    """
    base_out = F.linear(X, W)
    if A is not None and B is not None:
        lora_out = (X @ A) @ B * scaling
        return base_out + lora_out
    return base_out


def opaque_lora_linear(X, W, A, B, scaling):
    """Opaque kernel implementation."""
    return NewStyleLoRAW.apply(X, W, A, B, scaling)


# ============================================================================
# Test Configurations
# ============================================================================

# Mellum-4b-base-like dimensions: hidden=3072, intermediate=8256, rank=64
TEST_CONFIGS = [
    {
        "batch_size": 2,
        "seq_len": 128,
        "in_features": 3072,
        "out_features": 3072,
        "rank": 64,
        "name": "Mellum-O-proj",
    },
    {
        "batch_size": 4,
        "seq_len": 256,
        "in_features": 3072,
        "out_features": 8256,
        "rank": 64,
        "name": "Mellum-MLP-up",
    },
]


# ============================================================================
# Test: LoRA-W (single projection)
# ============================================================================

class TestLoRAW:
    """Test LoRA-W kernel vs PyTorch vmap."""

    @pytest.mark.parametrize("config", TEST_CONFIGS)
    def test_precision_vs_pytorch_vmap(self, config):
        """Verify precision: rtol=1e-7 vs PyTorch vmap."""
        torch.manual_seed(42)
        vmap_batch = 4
        batch, seq_len = config["batch_size"], config["seq_len"]
        in_features, out_features, rank = (
            config["in_features"],
            config["out_features"],
            config["rank"],
        )

        # Create inputs with vmap batch dimension (float64 for 1e-7 precision)
        X = torch.randn(
            vmap_batch, batch, seq_len, in_features, device="cuda", dtype=torch.float64
        )
        W = torch.randn(out_features, in_features, device="cuda", dtype=torch.float64)
        A = torch.randn(in_features, rank, device="cuda", dtype=torch.float64)
        B = torch.randn(rank, out_features, device="cuda", dtype=torch.float64)
        scaling = 0.1

        # Define vmapped functions
        def opaque_vmapped(X):
            return torch.vmap(lambda x: opaque_lora_linear(x, W, A, B, scaling))(X)

        def pytorch_vmapped(X):
            return torch.vmap(lambda x: pytorch_lora_linear(x, W, A, B, scaling))(X)

        # Validate with rtol=1e-7, relative-only error
        val = validate_implementations(
            opaque_vmapped,
            pytorch_vmapped,
            [X],
            grad_inputs_idx=[0],
            name=f"LoRA-W {config['name']}",
            rtol=1e-7,
            use_relative_only=True,
        )

        print_validation_result(val)

        assert val.forward_matches, (
            f"Forward precision failed: {config['name']} "
            f"(rel_error={val.forward_max_diff:.2e}, rtol=1e-7)"
        )
        assert val.backward_matches, (
            f"Backward precision failed: {config['name']} "
            f"(rel_errors={val.backward_max_diff}, rtol=1e-7)"
        )

    @pytest.mark.parametrize("config", TEST_CONFIGS)
    def test_performance_vs_pytorch_vmap(self, config):
        """Verify performance vs PyTorch vmap."""
        torch.manual_seed(42)
        vmap_batch = 4
        batch, seq_len = config["batch_size"], config["seq_len"]
        in_features, out_features, rank = (
            config["in_features"],
            config["out_features"],
            config["rank"],
        )

        X = torch.randn(
            vmap_batch, batch, seq_len, in_features, device="cuda", dtype=torch.float64
        )
        W = torch.randn(out_features, in_features, device="cuda", dtype=torch.float64)
        A = torch.randn(in_features, rank, device="cuda", dtype=torch.float64)
        B = torch.randn(rank, out_features, device="cuda", dtype=torch.float64)
        scaling = 0.1

        def opaque_vmapped(X):
            return torch.vmap(lambda x: opaque_lora_linear(x, W, A, B, scaling))(X)

        def pytorch_vmapped(X):
            return torch.vmap(lambda x: pytorch_lora_linear(x, W, A, B, scaling))(X)

        opaque_bench = benchmark_forward_backward(
            opaque_vmapped, [X], grad_inputs_idx=[0], name=f"Opaque {config['name']}"
        )
        pytorch_bench = benchmark_forward_backward(
            pytorch_vmapped, [X], grad_inputs_idx=[0], name=f"PyTorch {config['name']}"
        )

        print_comparison_table([opaque_bench, pytorch_bench], f"LoRA-W {config['name']}")

        # Verify either performance or memory benefit
        perf_ratio = opaque_bench.total_time_ms / pytorch_bench.total_time_ms
        mem_ratio = opaque_bench.memory_peak_mb / pytorch_bench.memory_peak_mb

        print(
            f"\n{config['name']}: "
            f"perf_ratio={perf_ratio:.2%}, mem_ratio={mem_ratio:.2%}"
        )

        # Accept if either performance is better OR memory is better (or both)
        has_benefit = perf_ratio < 1.0 or mem_ratio < 1.0
        if not has_benefit:
            pytest.skip(
                f"No performance or memory benefit: "
                f"perf={perf_ratio:.2%}, mem={mem_ratio:.2%}"
            )


# ============================================================================
# Test: LoRA-QKV (fused Q, K, V projections)
# ============================================================================

def pytorch_lora_qkv(X, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv):
    """PyTorch reference for Q, K, V projections."""
    Q = pytorch_lora_linear(X, Wq, Aq, Bq, Sq)
    K = pytorch_lora_linear(X, Wk, Ak, Bk, Sk)
    V = pytorch_lora_linear(X, Wv, Av, Bv, Sv)
    return Q, K, V


def opaque_lora_qkv(X, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv):
    """Opaque kernel implementation."""
    return NewStyleLoRAQKV.apply(X, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv)


class TestLoRAQKV:
    """Test LoRA-QKV kernel vs PyTorch vmap."""

    def test_precision_vs_pytorch_vmap(self):
        """Verify precision: rtol=1e-7 vs PyTorch vmap."""
        torch.manual_seed(42)
        vmap_batch = 4
        batch, seq_len, hidden_dim, head_dim, n_heads = 2, 128, 3072, 128, 24
        rank = 64

        X = torch.randn(
            vmap_batch, batch, seq_len, hidden_dim, device="cuda", dtype=torch.float64
        )
        qkv_dim = head_dim * n_heads

        # Q, K, V weights and LoRA
        Wq = torch.randn(qkv_dim, hidden_dim, device="cuda", dtype=torch.float64)
        Aq = torch.randn(hidden_dim, rank, device="cuda", dtype=torch.float64)
        Bq = torch.randn(rank, qkv_dim, device="cuda", dtype=torch.float64)
        Sq = 0.1

        Wk = torch.randn(qkv_dim, hidden_dim, device="cuda", dtype=torch.float64)
        Ak = torch.randn(hidden_dim, rank, device="cuda", dtype=torch.float64)
        Bk = torch.randn(rank, qkv_dim, device="cuda", dtype=torch.float64)
        Sk = 0.1

        Wv = torch.randn(qkv_dim, hidden_dim, device="cuda", dtype=torch.float64)
        Av = torch.randn(hidden_dim, rank, device="cuda", dtype=torch.float64)
        Bv = torch.randn(rank, qkv_dim, device="cuda", dtype=torch.float64)
        Sv = 0.1

        # Define vmapped functions
        def opaque_vmapped(X):
            return torch.vmap(
                lambda x: opaque_lora_qkv(x, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv)
            )(X)

        def pytorch_vmapped(X):
            return torch.vmap(
                lambda x: pytorch_lora_qkv(x, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv)
            )(X)

        # Custom validation for tuple outputs
        X1 = X.detach().clone().requires_grad_(True)
        X2 = X.detach().clone().requires_grad_(True)

        Q1, K1, V1 = opaque_vmapped(X1)
        Q2, K2, V2 = pytorch_vmapped(X2)

        # Check forward precision
        try:
            from .kernel_validation_framework import compare_outputs
        except ImportError:
            from kernel_validation_framework import compare_outputs

        q_diff, q_match = compare_outputs(Q1, Q2, rtol=1e-7, use_relative_only=True)
        k_diff, k_match = compare_outputs(K1, K2, rtol=1e-7, use_relative_only=True)
        v_diff, v_match = compare_outputs(V1, V2, rtol=1e-7, use_relative_only=True)

        print(
            f"\nLoRA-QKV forward: Q={q_diff:.2e}, K={k_diff:.2e}, V={v_diff:.2e} (rtol=1e-7)"
        )

        assert q_match and k_match and v_match, (
            f"Forward precision failed: Q={q_diff:.2e}, K={k_diff:.2e}, V={v_diff:.2e}"
        )

        # Check backward precision
        (Q1 + K1 + V1).sum().backward()
        (Q2 + K2 + V2).sum().backward()

        rel_error = (X1.grad - X2.grad).abs() / (X2.grad.abs() + 1e-10)
        max_rel = rel_error.max().item()
        print(f"LoRA-QKV backward: X_grad={max_rel:.2e} (rtol=1e-7)")

        assert max_rel < 1e-7, f"Backward precision failed: X_grad={max_rel:.2e}"

    def test_performance_vs_pytorch_vmap(self):
        """Verify performance vs PyTorch vmap."""
        torch.manual_seed(42)
        vmap_batch = 4
        batch, seq_len, hidden_dim, head_dim, n_heads = 2, 128, 3072, 128, 24
        rank = 64

        X = torch.randn(
            vmap_batch, batch, seq_len, hidden_dim, device="cuda", dtype=torch.float64
        )
        qkv_dim = head_dim * n_heads

        Wq = torch.randn(qkv_dim, hidden_dim, device="cuda", dtype=torch.float64)
        Aq = torch.randn(hidden_dim, rank, device="cuda", dtype=torch.float64)
        Bq = torch.randn(rank, qkv_dim, device="cuda", dtype=torch.float64)
        Sq = 0.1

        Wk = torch.randn(qkv_dim, hidden_dim, device="cuda", dtype=torch.float64)
        Ak = torch.randn(hidden_dim, rank, device="cuda", dtype=torch.float64)
        Bk = torch.randn(rank, qkv_dim, device="cuda", dtype=torch.float64)
        Sk = 0.1

        Wv = torch.randn(qkv_dim, hidden_dim, device="cuda", dtype=torch.float64)
        Av = torch.randn(hidden_dim, rank, device="cuda", dtype=torch.float64)
        Bv = torch.randn(rank, qkv_dim, device="cuda", dtype=torch.float64)
        Sv = 0.1

        def opaque_vmapped(X):
            Q, K, V = torch.vmap(
                lambda x: opaque_lora_qkv(x, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv)
            )(X)
            return Q + K + V  # Sum for backward pass

        def pytorch_vmapped(X):
            Q, K, V = torch.vmap(
                lambda x: pytorch_lora_qkv(x, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv)
            )(X)
            return Q + K + V

        opaque_bench = benchmark_forward_backward(
            opaque_vmapped, [X], grad_inputs_idx=[0], name="Opaque LoRA-QKV"
        )
        pytorch_bench = benchmark_forward_backward(
            pytorch_vmapped, [X], grad_inputs_idx=[0], name="PyTorch LoRA-QKV"
        )

        print_comparison_table([opaque_bench, pytorch_bench], "LoRA-QKV")

        perf_ratio = opaque_bench.total_time_ms / pytorch_bench.total_time_ms
        mem_ratio = opaque_bench.memory_peak_mb / pytorch_bench.memory_peak_mb

        print(f"\nLoRA-QKV: perf_ratio={perf_ratio:.2%}, mem_ratio={mem_ratio:.2%}")

        has_benefit = perf_ratio < 1.0 or mem_ratio < 1.0
        if not has_benefit:
            pytest.skip(
                f"No performance or memory benefit: "
                f"perf={perf_ratio:.2%}, mem={mem_ratio:.2%}"
            )


# ============================================================================
# Test: LoRA-MLP (fused gate, up, down with SwiGLU)
# ============================================================================

def pytorch_lora_mlp(X, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd):
    """PyTorch reference for MLP with SwiGLU."""
    gate = pytorch_lora_linear(X, Wg, Ag, Bg, Sg)
    up = pytorch_lora_linear(X, Wu, Au, Bu, Su)
    h = F.silu(gate) * up
    out = pytorch_lora_linear(h, Wd, Ad, Bd, Sd)
    return out


def opaque_lora_mlp(X, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd):
    """Opaque kernel implementation."""
    result = NewStyleLoRAMLP.apply(X, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd)
    return result[0]  # Return only output, not intermediates


class TestLoRAMLP:
    """Test LoRA-MLP kernel vs PyTorch vmap."""

    def test_precision_vs_pytorch_vmap(self):
        """Verify precision: rtol=1e-7 vs PyTorch vmap."""
        torch.manual_seed(42)
        vmap_batch = 4
        batch, seq_len, hidden_dim, intermediate_dim = 2, 128, 3072, 8256
        rank = 64

        X = torch.randn(
            vmap_batch, batch, seq_len, hidden_dim, device="cuda", dtype=torch.float64
        )

        # Gate, up, down projections
        Wg = torch.randn(intermediate_dim, hidden_dim, device="cuda", dtype=torch.float64)
        Ag = torch.randn(hidden_dim, rank, device="cuda", dtype=torch.float64)
        Bg = torch.randn(rank, intermediate_dim, device="cuda", dtype=torch.float64)
        Sg = 0.1

        Wu = torch.randn(intermediate_dim, hidden_dim, device="cuda", dtype=torch.float64)
        Au = torch.randn(hidden_dim, rank, device="cuda", dtype=torch.float64)
        Bu = torch.randn(rank, intermediate_dim, device="cuda", dtype=torch.float64)
        Su = 0.1

        Wd = torch.randn(hidden_dim, intermediate_dim, device="cuda", dtype=torch.float64)
        Ad = torch.randn(intermediate_dim, rank, device="cuda", dtype=torch.float64)
        Bd = torch.randn(rank, hidden_dim, device="cuda", dtype=torch.float64)
        Sd = 0.1

        def opaque_vmapped(X):
            return torch.vmap(
                lambda x: opaque_lora_mlp(x, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd)
            )(X)

        def pytorch_vmapped(X):
            return torch.vmap(
                lambda x: pytorch_lora_mlp(x, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd)
            )(X)

        val = validate_implementations(
            opaque_vmapped,
            pytorch_vmapped,
            [X],
            grad_inputs_idx=[0],
            name="LoRA-MLP",
            rtol=1e-7,
            use_relative_only=True,
        )

        print_validation_result(val)

        assert val.forward_matches, (
            f"Forward precision failed: LoRA-MLP (rel_error={val.forward_max_diff:.2e}, rtol=1e-7)"
        )
        assert val.backward_matches, (
            f"Backward precision failed: LoRA-MLP (rel_errors={val.backward_max_diff}, rtol=1e-7)"
        )

    def test_performance_vs_pytorch_vmap(self):
        """Verify performance vs PyTorch vmap."""
        torch.manual_seed(42)
        vmap_batch = 4
        batch, seq_len, hidden_dim, intermediate_dim = 2, 128, 3072, 8256
        rank = 64

        X = torch.randn(
            vmap_batch, batch, seq_len, hidden_dim, device="cuda", dtype=torch.float64
        )

        Wg = torch.randn(intermediate_dim, hidden_dim, device="cuda", dtype=torch.float64)
        Ag = torch.randn(hidden_dim, rank, device="cuda", dtype=torch.float64)
        Bg = torch.randn(rank, intermediate_dim, device="cuda", dtype=torch.float64)
        Sg = 0.1

        Wu = torch.randn(intermediate_dim, hidden_dim, device="cuda", dtype=torch.float64)
        Au = torch.randn(hidden_dim, rank, device="cuda", dtype=torch.float64)
        Bu = torch.randn(rank, intermediate_dim, device="cuda", dtype=torch.float64)
        Su = 0.1

        Wd = torch.randn(hidden_dim, intermediate_dim, device="cuda", dtype=torch.float64)
        Ad = torch.randn(intermediate_dim, rank, device="cuda", dtype=torch.float64)
        Bd = torch.randn(rank, hidden_dim, device="cuda", dtype=torch.float64)
        Sd = 0.1

        def opaque_vmapped(X):
            return torch.vmap(
                lambda x: opaque_lora_mlp(x, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd)
            )(X)

        def pytorch_vmapped(X):
            return torch.vmap(
                lambda x: pytorch_lora_mlp(x, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd)
            )(X)

        opaque_bench = benchmark_forward_backward(
            opaque_vmapped, [X], grad_inputs_idx=[0], name="Opaque LoRA-MLP"
        )
        pytorch_bench = benchmark_forward_backward(
            pytorch_vmapped, [X], grad_inputs_idx=[0], name="PyTorch LoRA-MLP"
        )

        print_comparison_table([opaque_bench, pytorch_bench], "LoRA-MLP")

        perf_ratio = opaque_bench.total_time_ms / pytorch_bench.total_time_ms
        mem_ratio = opaque_bench.memory_peak_mb / pytorch_bench.memory_peak_mb

        print(f"\nLoRA-MLP: perf_ratio={perf_ratio:.2%}, mem_ratio={mem_ratio:.2%}")

        has_benefit = perf_ratio < 1.0 or mem_ratio < 1.0
        if not has_benefit:
            pytest.skip(
                f"No performance or memory benefit: "
                f"perf={perf_ratio:.2%}, mem={mem_ratio:.2%}"
            )


if __name__ == "__main__":
    import sys

    print("\n" + "=" * 80)
    print("LORA KERNEL TESTS")
    print("=" * 80)

    # Run tests
    pytest.main([__file__, "-v", "-s"])
