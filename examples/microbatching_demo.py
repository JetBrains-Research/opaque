"""Microbatching demonstration for memory-efficient DP-SGD.

This example shows how to use the microbatch_size parameter in clipped_grad()
to reduce memory usage when training with large batches. Microbatching is
particularly important for DP-SGD because computing per-example gradients
materializes gradients for the entire batch.

Key concepts:
1. Microbatching processes the batch in smaller chunks (microbatches)
2. Gradients are computed and clipped per microbatch, then accumulated
3. Memory usage scales with microbatch_size, not batch_size
4. Results are numerically identical to processing the full batch at once

This is the recommended alternative to gradient checkpointing (which is
incompatible with vmap). See docs/development/GRADIENT_CHECKPOINTING_PLAN.md
for technical details.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from opaque import clipped_grad, gaussian


def measure_memory_usage():
    """Utility to measure current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        return allocated, reserved
    return 0.0, 0.0


def demo_basic_microbatching():
    """Demonstrate basic microbatching usage."""
    print("=" * 70)
    print("DEMO 1: Basic Microbatching")
    print("=" * 70)

    # Create a simple linear model with synthetic data
    torch.manual_seed(42)
    input_dim = 50
    n_samples = 256
    batch_size = 128  # Large batch for DP

    X = torch.randn(n_samples, input_dim)
    y = torch.randn(n_samples)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    params = torch.randn(input_dim, requires_grad=False)

    def loss_fn(params, x, y):
        pred = x @ params
        return ((pred - y) ** 2).sum()

    print(f"\nSetup:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Batch size: {batch_size}")
    print(f"  Dataset size: {n_samples}")

    # Get a batch for testing
    batch_x, batch_y = next(iter(dataloader))

    # Method 1: Without microbatching (processes entire batch at once)
    print(f"\n1. WITHOUT Microbatching:")
    print(f"   - Processes all {batch_size} examples simultaneously")
    print(f"   - Higher memory usage")

    grad_fn_full, clip_state_full = clipped_grad(
        loss_fn,
        argnums=0,
        batch_argnums=(1, 2),
        l2_clip_norm=1.0,
        microbatch_size=None,  # No microbatching
    )

    mem_before, _ = measure_memory_usage()
    grads_full, _ = grad_fn_full(params, batch_x, batch_y, state=clip_state_full)
    mem_after, _ = measure_memory_usage()
    memory_full = mem_after - mem_before

    print(f"   - Memory increase: ~{memory_full:.2f} MB")
    print(f"   - Gradient sum (first 5): {grads_full[:5]}")

    # Method 2: With microbatching (processes in smaller chunks)
    microbatch_size = 32
    print(f"\n2. WITH Microbatching (microbatch_size={microbatch_size}):")
    print(f"   - Processes {batch_size // microbatch_size} chunks of {microbatch_size} examples")
    print(f"   - Lower memory usage")

    grad_fn_micro, clip_state_micro = clipped_grad(
        loss_fn,
        argnums=0,
        batch_argnums=(1, 2),
        l2_clip_norm=1.0,
        microbatch_size=microbatch_size,  # Process 32 examples at a time
    )

    mem_before, _ = measure_memory_usage()
    grads_micro, _ = grad_fn_micro(params, batch_x, batch_y, state=clip_state_micro)
    mem_after, _ = measure_memory_usage()
    memory_micro = mem_after - mem_before

    print(f"   - Memory increase: ~{memory_micro:.2f} MB")
    print(f"   - Gradient sum (first 5): {grads_micro[:5]}")

    # Verify numerical equivalence
    print(f"\n3. Verification:")
    are_close = torch.allclose(grads_full, grads_micro, atol=1e-5)
    max_diff = torch.max(torch.abs(grads_full - grads_micro)).item()
    print(f"   - Results identical? {are_close}")
    print(f"   - Max difference: {max_diff:.2e}")
    print(f"   - Memory saved: ~{memory_full - memory_micro:.2f} MB ({(1 - memory_micro/max(memory_full, 0.001))*100:.1f}%)")

    assert are_close, "Microbatching should produce identical results!"
    print("\n✅ Microbatching works correctly and saves memory!")


def demo_full_training():
    """Demonstrate microbatching in a full DP-SGD training loop."""
    print("\n" + "=" * 70)
    print("DEMO 2: Full DP-SGD Training with Microbatching")
    print("=" * 70)

    # Hyperparameters
    input_dim = 30
    output_dim = 1
    n_samples = 500
    batch_size = 64  # Large batch for DP
    microbatch_size = 16  # Process 16 examples at a time
    epochs = 3
    lr = 0.01
    l2_clip_norm = 1.0
    noise_multiplier = 1.1

    print(f"\nHyperparameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Microbatch size: {microbatch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  L2 clip norm: {l2_clip_norm}")
    print(f"  Noise multiplier: {noise_multiplier}")

    # Generate synthetic data
    torch.manual_seed(42)
    X = torch.randn(n_samples, input_dim)
    true_weights = torch.randn(input_dim)
    y = X @ true_weights + torch.randn(n_samples) * 0.1

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model parameters
    params = torch.randn(input_dim, requires_grad=False)

    # Define loss function
    def loss_fn(params, x, y):
        pred = x @ params
        return ((pred - y) ** 2).mean()

    # Create DP-SGD components with microbatching
    print(f"\nInitializing DP-SGD with microbatching...")
    grad_fn, clip_state = clipped_grad(
        loss_fn,
        argnums=0,
        batch_argnums=(1, 2),
        l2_clip_norm=l2_clip_norm,
        normalize_by=batch_size,  # Normalize by batch size for gradient averaging
        microbatch_size=microbatch_size,  # KEY PARAMETER: Enable microbatching
    )

    noise_fn = gaussian(stddev=noise_multiplier * grad_fn.clip_norm)

    print(f"DP-SGD ready! Clip norm: {grad_fn.clip_norm:.2f}")
    print(f"\nTraining...")

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
            # Compute clipped gradients (with microbatching!)
            grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

            # Add DP noise
            noisy_grads = noise_fn(grads)

            # Update parameters
            params = params - lr * noisy_grads

            # Track loss (for monitoring only, not part of DP-SGD)
            with torch.no_grad():
                loss = loss_fn(params, batch_x, batch_y)
                epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"  Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.4f}")

    print("\n✅ Training complete with microbatching!")
    print("   Memory usage was controlled by microbatch_size, not batch_size")


def demo_choosing_microbatch_size():
    """Demonstrate how to choose appropriate microbatch size."""
    print("\n" + "=" * 70)
    print("DEMO 3: Choosing the Right Microbatch Size")
    print("=" * 70)

    torch.manual_seed(42)
    input_dim = 40
    batch_size = 128

    X = torch.randn(batch_size, input_dim)
    y = torch.randn(batch_size)
    params = torch.randn(input_dim, requires_grad=False)

    def loss_fn(params, x, y):
        pred = x @ params
        return ((pred - y) ** 2).sum()

    print(f"\nTesting different microbatch sizes (batch_size={batch_size}):")
    print(f"\n{'Microbatch Size':<18} {'Memory (MB)':<15} {'Time (ms)':<12} {'Status'}")
    print("-" * 70)

    # Test various microbatch sizes
    microbatch_sizes = [None, 64, 32, 16, 8, 4, 1]

    for mb_size in microbatch_sizes:
        grad_fn, clip_state = clipped_grad(
            loss_fn,
            argnums=0,
            batch_argnums=(1, 2),
            l2_clip_norm=1.0,
            microbatch_size=mb_size,
        )

        # Warmup
        _, _ = grad_fn(params, X, y, state=clip_state)

        # Measure
        mem_before, _ = measure_memory_usage()
        import time
        start = time.time()
        _, _ = grad_fn(params, X, y, state=clip_state)
        elapsed = (time.time() - start) * 1000  # ms
        mem_after, _ = measure_memory_usage()
        memory = mem_after - mem_before

        mb_str = "Full batch" if mb_size is None else str(mb_size)
        status = "✓ Recommended" if mb_size == 32 else ""
        print(f"{mb_str:<18} {memory:>10.2f} MB   {elapsed:>8.1f} ms   {status}")

    print("\nGuidelines for choosing microbatch_size:")
    print("  • Start with batch_size // 4 (e.g., 32 for batch_size=128)")
    print("  • Increase if you have memory to spare (faster)")
    print("  • Decrease if you hit OOM errors (slower but more memory-efficient)")
    print("  • microbatch_size=1 is slowest but most memory-efficient")
    print("  • Setting to None processes full batch (highest memory, baseline speed)")


def main():
    """Run all microbatching demonstrations."""
    print("\n" + "=" * 70)
    print("MICROBATCHING TUTORIAL FOR DP-SGD")
    print("=" * 70)
    print("\nMicrobatching is Opaque's solution for memory-efficient DP-SGD training.")
    print("It's the recommended alternative to gradient checkpointing (which doesn't")
    print("work with vmap). See docs/development/GRADIENT_CHECKPOINTING_PLAN.md")
    print("")

    # Run demos
    demo_basic_microbatching()
    demo_full_training()
    demo_choosing_microbatch_size()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key Takeaways:
1. ✅ Microbatching is already implemented in Opaque
2. 🎯 Use microbatch_size parameter in clipped_grad() to enable it
3. 💾 Memory usage scales with microbatch_size, not batch_size
4. 🔬 Results are numerically identical to full-batch processing
5. 📈 Typical recommendation: microbatch_size = batch_size // 4

Example usage:
    grad_fn, clip_state = clipped_grad(
        loss_fn,
        l2_clip_norm=1.0,
        microbatch_size=32,  # <-- Enable microbatching here
    )

For more details, see:
- docs/development/GRADIENT_CHECKPOINTING_PLAN.md (technical analysis)
- docs/development/GRADIENT_CHECKPOINTING_SUMMARY.md (quick reference)
- examples/train_causal_lm.py (real-world usage with LLMs)
""")


if __name__ == "__main__":
    main()
