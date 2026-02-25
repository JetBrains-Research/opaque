"""
Cross Entropy Kernel Validation

3-phase validation:
1. Opaque kernel vs PyTorch F.cross_entropy (baseline)
2. Opaque kernel backward vs PyTorch backward
3. Opaque vmap vs torch.vmap(PyTorch)

Note: Unsloth comparison is optional since it has many dependencies.
"""

import pytest
import torch
import torch.nn.functional as F
import sys
import time
import gc

try:
    from .kernel_validation_framework import (
        benchmark_forward_backward,
        validate_implementations,
        print_benchmark_result,
        print_validation_result,
        print_comparison_table,
    )
except ImportError:
    from kernel_validation_framework import (
        benchmark_forward_backward,
        validate_implementations,
        print_benchmark_result,
        print_validation_result,
        print_comparison_table,
    )


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for kernel tests"
)


# ============================================================================
# Test Configurations
# ============================================================================

TEST_CONFIGS = [
    {"batch_size": 4, "seq_len": 128, "vocab_size": 32000, "name": "Llama-style"},
    {"batch_size": 2, "seq_len": 256, "vocab_size": 50257, "name": "GPT2-style"},
    {"batch_size": 1, "seq_len": 512, "vocab_size": 32000, "name": "Long-seq"},
]


# ============================================================================
# Reference Implementations
# ============================================================================

def pytorch_cross_entropy(logits, labels):
    """PyTorch reference implementation."""
    batch_seq = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)
    loss = F.cross_entropy(logits_flat, labels_flat, reduction="mean")
    return loss


def opaque_cross_entropy(logits, labels):
    """Opaque kernel implementation."""
    from opaque.kernels import NewStyleCrossEntropy
    losses, _ = NewStyleCrossEntropy.apply(logits, labels)
    # For vmap compatibility, avoid data-dependent control flow
    mask = (labels != -100).float()
    n_valid = mask.sum()
    masked_losses = losses * mask
    return masked_losses.sum() / torch.clamp(n_valid, min=1.0)


# ============================================================================
# Phase 1: Forward Pass Validation
# ============================================================================

class TestCrossEntropyForward:
    """Test forward pass matches PyTorch."""

    @pytest.mark.parametrize("config", TEST_CONFIGS)
    def test_forward_matches_pytorch(self, config):
        """Verify forward pass matches PyTorch."""
        torch.manual_seed(42)
        batch, seq_len, vocab = config["batch_size"], config["seq_len"], config["vocab_size"]

        logits = torch.randn(batch, seq_len, vocab, device="cuda", dtype=torch.float32)
        labels = torch.randint(0, vocab, (batch, seq_len), device="cuda")

        # Opaque kernel
        opaque_loss = opaque_cross_entropy(logits, labels)

        # PyTorch reference
        pytorch_loss = pytorch_cross_entropy(logits, labels)

        max_diff = (opaque_loss - pytorch_loss).abs().item()
        print(f"\n{config['name']}: forward diff = {max_diff:.2e}")

        assert torch.allclose(opaque_loss, pytorch_loss, rtol=1e-4, atol=1e-4), (
            f"Forward mismatch for {config['name']}: diff={max_diff:.2e}"
        )


# ============================================================================
# Phase 2: Backward Pass Validation
# ============================================================================

class TestCrossEntropyBackward:
    """Test backward pass matches PyTorch."""

    @pytest.mark.parametrize("config", TEST_CONFIGS)
    def test_backward_matches_pytorch(self, config):
        """Verify gradients match PyTorch."""
        torch.manual_seed(42)
        batch, seq_len, vocab = config["batch_size"], config["seq_len"], config["vocab_size"]

        # Opaque kernel
        logits_opaque = torch.randn(
            batch, seq_len, vocab, device="cuda", dtype=torch.float32, requires_grad=True
        )
        labels = torch.randint(0, vocab, (batch, seq_len), device="cuda")

        opaque_loss = opaque_cross_entropy(logits_opaque, labels)
        opaque_loss.backward()
        opaque_grad = logits_opaque.grad.clone()

        # PyTorch reference
        logits_pytorch = logits_opaque.detach().clone().requires_grad_(True)
        pytorch_loss = pytorch_cross_entropy(logits_pytorch, labels)
        pytorch_loss.backward()
        pytorch_grad = logits_pytorch.grad

        max_diff = (opaque_grad - pytorch_grad).abs().max().item()
        print(f"\n{config['name']}: backward max diff = {max_diff:.2e}")

        assert torch.allclose(opaque_grad, pytorch_grad, rtol=1e-4, atol=1e-4), (
            f"Backward mismatch for {config['name']}: max_diff={max_diff:.2e}"
        )


# ============================================================================
# Phase 3: Vmap Validation
# ============================================================================

class TestCrossEntropyVmap:
    """Test vmap support for DP-SGD."""

    def test_vmap_forward_matches_loop(self):
        """Verify vmap produces same results as loop."""
        torch.manual_seed(42)
        vmap_batch, batch, seq_len, vocab = 4, 2, 64, 1000

        logits = torch.randn(
            vmap_batch, batch, seq_len, vocab, device="cuda", dtype=torch.float32
        )
        labels = torch.randint(0, vocab, (vmap_batch, batch, seq_len), device="cuda")

        # vmap over opaque kernel
        vmapped_fn = torch.vmap(opaque_cross_entropy, in_dims=(0, 0))
        vmap_results = vmapped_fn(logits, labels)

        # Loop reference
        loop_results = torch.stack([
            opaque_cross_entropy(logits[i], labels[i])
            for i in range(vmap_batch)
        ])

        max_diff = (vmap_results - loop_results).abs().max().item()
        print(f"\nvmap vs loop: max diff = {max_diff:.2e}")

        assert torch.allclose(vmap_results, loop_results, rtol=1e-5, atol=1e-5), (
            f"vmap mismatch: max_diff={max_diff:.2e}"
        )

    def test_vmap_matches_pytorch_vmap(self):
        """Verify Opaque vmap matches torch.vmap(PyTorch)."""
        torch.manual_seed(42)
        vmap_batch, batch, seq_len, vocab = 4, 2, 64, 1000

        logits = torch.randn(
            vmap_batch, batch, seq_len, vocab, device="cuda", dtype=torch.float32
        )
        labels = torch.randint(0, vocab, (vmap_batch, batch, seq_len), device="cuda")

        # Opaque vmap
        opaque_vmapped = torch.vmap(opaque_cross_entropy, in_dims=(0, 0))
        opaque_results = opaque_vmapped(logits, labels)

        # PyTorch vmap
        pytorch_vmapped = torch.vmap(pytorch_cross_entropy, in_dims=(0, 0))
        pytorch_results = pytorch_vmapped(logits, labels)

        max_diff = (opaque_results - pytorch_results).abs().max().item()
        print(f"\nOpaque vmap vs PyTorch vmap: max diff = {max_diff:.2e}")

        assert torch.allclose(opaque_results, pytorch_results, rtol=1e-4, atol=1e-4), (
            f"vmap mismatch: max_diff={max_diff:.2e}"
        )

    def test_vmap_backward(self):
        """Verify vmap backward pass works correctly."""
        torch.manual_seed(42)
        vmap_batch, batch, seq_len, vocab = 3, 2, 32, 500

        logits = torch.randn(
            vmap_batch, batch, seq_len, vocab, device="cuda", dtype=torch.float32, requires_grad=True
        )
        labels = torch.randint(0, vocab, (vmap_batch, batch, seq_len), device="cuda")

        # vmap forward + backward
        vmapped_fn = torch.vmap(opaque_cross_entropy, in_dims=(0, 0))
        vmap_results = vmapped_fn(logits, labels)
        vmap_results.sum().backward()

        assert logits.grad is not None, "No gradient computed"
        assert logits.grad.shape == logits.shape, "Gradient shape mismatch"
        print(f"\nvmap backward: grad shape = {logits.grad.shape}")


# ============================================================================
# Performance Benchmarks
# ============================================================================

class TestCrossEntropyPerformance:
    """Performance benchmarks."""

    def test_benchmark_forward_backward(self):
        """Benchmark forward and backward pass."""
        torch.manual_seed(42)
        batch, seq_len, vocab = 8, 256, 32000

        logits = torch.randn(batch, seq_len, vocab, device="cuda", dtype=torch.float32)
        labels = torch.randint(0, vocab, (batch, seq_len), device="cuda")

        results = []

        # PyTorch
        bench_pytorch = benchmark_forward_backward(
            lambda l, t: pytorch_cross_entropy(l, t),
            [logits, labels], [0], name="PyTorch F.cross_entropy"
        )
        results.append(bench_pytorch)

        # Opaque
        bench_opaque = benchmark_forward_backward(
            lambda l, t: opaque_cross_entropy(l, t),
            [logits, labels], [0], name="Opaque kernel"
        )
        results.append(bench_opaque)

        print_comparison_table(results, "Cross Entropy Performance (batch=8, seq=256, vocab=32K)")

    def test_benchmark_vmap(self):
        """Benchmark vmap performance."""
        torch.manual_seed(42)
        vmap_batch, batch, seq_len, vocab = 8, 4, 128, 32000

        logits = torch.randn(
            vmap_batch, batch, seq_len, vocab, device="cuda", dtype=torch.float32
        )
        labels = torch.randint(0, vocab, (vmap_batch, batch, seq_len), device="cuda")

        results = []

        # PyTorch vmap
        pytorch_vmapped = torch.vmap(pytorch_cross_entropy, in_dims=(0, 0))
        bench_pytorch_vmap = benchmark_forward_backward(
            lambda l, t: pytorch_vmapped(l, t),
            [logits, labels], [0], name="PyTorch vmap"
        )
        results.append(bench_pytorch_vmap)

        # Opaque vmap
        opaque_vmapped = torch.vmap(opaque_cross_entropy, in_dims=(0, 0))
        bench_opaque_vmap = benchmark_forward_backward(
            lambda l, t: opaque_vmapped(l, t),
            [logits, labels], [0], name="Opaque vmap"
        )
        results.append(bench_opaque_vmap)

        print_comparison_table(results, f"Cross Entropy vmap Performance (vmap_batch={vmap_batch})")


# ============================================================================
# Full Validation Suite
# ============================================================================

def run_full_validation():
    """Run complete validation suite with detailed output."""
    print("\n" + "="*80)
    print("CROSS ENTROPY KERNEL VALIDATION")
    print("="*80)

    all_passed = True
    results = {}

    # Phase 1: Forward validation
    print("\n--- Phase 1: Forward Pass Validation ---")
    for config in TEST_CONFIGS:
        torch.manual_seed(42)
        batch, seq_len, vocab = config["batch_size"], config["seq_len"], config["vocab_size"]

        logits = torch.randn(batch, seq_len, vocab, device="cuda", dtype=torch.float32)
        labels = torch.randint(0, vocab, (batch, seq_len), device="cuda")

        opaque_loss = opaque_cross_entropy(logits, labels)
        pytorch_loss = pytorch_cross_entropy(logits, labels)

        max_diff = (opaque_loss - pytorch_loss).abs().item()
        passed = max_diff < 1e-4
        status = "PASS" if passed else "FAIL"
        print(f"  {config['name']}: {status} (diff={max_diff:.2e})")

        if not passed:
            all_passed = False

    # Phase 2: Backward validation
    print("\n--- Phase 2: Backward Pass Validation ---")
    for config in TEST_CONFIGS:
        torch.manual_seed(42)
        batch, seq_len, vocab = config["batch_size"], config["seq_len"], config["vocab_size"]

        logits_opaque = torch.randn(
            batch, seq_len, vocab, device="cuda", dtype=torch.float32, requires_grad=True
        )
        labels = torch.randint(0, vocab, (batch, seq_len), device="cuda")

        opaque_loss = opaque_cross_entropy(logits_opaque, labels)
        opaque_loss.backward()
        opaque_grad = logits_opaque.grad.clone()

        logits_pytorch = logits_opaque.detach().clone().requires_grad_(True)
        pytorch_loss = pytorch_cross_entropy(logits_pytorch, labels)
        pytorch_loss.backward()
        pytorch_grad = logits_pytorch.grad

        max_diff = (opaque_grad - pytorch_grad).abs().max().item()
        passed = max_diff < 1e-3
        status = "PASS" if passed else "FAIL"
        print(f"  {config['name']}: {status} (max_diff={max_diff:.2e})")

        if not passed:
            all_passed = False

    # Phase 3: Vmap validation
    print("\n--- Phase 3: Vmap Validation ---")
    torch.manual_seed(42)
    vmap_batch, batch, seq_len, vocab = 4, 2, 64, 1000

    logits = torch.randn(
        vmap_batch, batch, seq_len, vocab, device="cuda", dtype=torch.float32
    )
    labels = torch.randint(0, vocab, (vmap_batch, batch, seq_len), device="cuda")

    opaque_vmapped = torch.vmap(opaque_cross_entropy, in_dims=(0, 0))
    pytorch_vmapped = torch.vmap(pytorch_cross_entropy, in_dims=(0, 0))

    opaque_results = opaque_vmapped(logits, labels)
    pytorch_results = pytorch_vmapped(logits, labels)

    max_diff = (opaque_results - pytorch_results).abs().max().item()
    passed = max_diff < 1e-4
    status = "PASS" if passed else "FAIL"
    print(f"  Opaque vmap vs PyTorch vmap: {status} (max_diff={max_diff:.2e})")

    if not passed:
        all_passed = False

    # Performance comparison
    print("\n--- Performance Comparison ---")
    batch, seq_len, vocab = 8, 256, 32000
    logits = torch.randn(batch, seq_len, vocab, device="cuda", dtype=torch.float32)
    labels = torch.randint(0, vocab, (batch, seq_len), device="cuda")

    bench_pytorch = benchmark_forward_backward(
        lambda l, t: pytorch_cross_entropy(l, t),
        [logits, labels], [0], name="PyTorch"
    )
    bench_opaque = benchmark_forward_backward(
        lambda l, t: opaque_cross_entropy(l, t),
        [logits, labels], [0], name="Opaque"
    )

    print_comparison_table([bench_pytorch, bench_opaque], "Forward+Backward Performance")

    # Summary
    print("\n" + "="*80)
    print(f"VALIDATION RESULT: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("="*80)

    return all_passed


if __name__ == "__main__":
    success = run_full_validation()
    sys.exit(0 if success else 1)
