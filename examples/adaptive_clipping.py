"""Example demonstrating adaptive clipping with explicit state-passing.

This example shows the functional API for adaptive gradient clipping where state is
passed explicitly as a parameter and returned as part of the output. This design
avoids mutable closures and works seamlessly with distributed training.
"""

import torch

from opaque.clipping import adaptive_clipped_grad


def simple_loss(params, x, y):
    """Simple linear regression loss."""
    pred = x @ params
    return ((pred - y) ** 2).mean()


def main():
    """Demonstrate adaptive clipping with explicit state-passing."""
    print("=" * 70)
    print("Adaptive Clipping Example")
    print("=" * 70)

    # Setup
    torch.manual_seed(42)
    dim = 5
    batch_size = 32
    num_steps = 100

    # Initialize model
    params = torch.randn(dim, requires_grad=False)
    print(f"\nModel: Linear regression with {dim} parameters")
    print(f"Batch size: {batch_size}")

    # Create adaptive clipping function with explicit state
    print("\nCreating adaptive clipping function...")
    grad_fn, clip_state = adaptive_clipped_grad(
        simple_loss,
        initial_clip_norm=0.1,  # Start low
        target_quantile=0.5,  # Target 50% clipping rate
        learning_rate=0.2,  # Adaptation speed
        batch_argnums=(1, 2),  # x and y are batched
    )

    print(f"Initial state:")
    print(f"  - clip_norm: {clip_state.clip_norm:.4f}")
    print(f"  - step: {clip_state.step}")
    print(f"  - clipping_rate: {clip_state.clipping_rate:.2%}")

    # Training loop
    print(f"\nRunning {num_steps} steps...")
    print("\nStep | Clip Norm | Clipping Rate | Sensitivity")
    print("-" * 55)

    for step in range(num_steps):
        # Generate batch
        batch_x = torch.randn(batch_size, dim)
        batch_y = torch.randn(batch_size)

        # Compute clipped gradients - state passed explicitly
        grad, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

        # Compute sensitivity for noise scaling (using state method)
        sens = clip_state.sensitivity()

        # Print progress
        if step % 10 == 0:
            print(
                f"{step:4d} | {clip_state.clip_norm:9.4f} | "
                f"{clip_state.clipping_rate:13.2%} | {sens:11.4f}"
            )

        # Simulate parameter update (simplified - no noise or optimizer here)
        # params = params - 0.01 * grad

    print("-" * 55)
    print(f"\nFinal state:")
    print(f"  - clip_norm: {clip_state.clip_norm:.4f}")
    print(f"  - step: {clip_state.step}")
    print(f"  - clipping_rate: {clip_state.clipping_rate:.2%}")

    print("\n" + "=" * 70)
    print("Key Benefits of Explicit State-Passing:")
    print("=" * 70)
    print("✓ State is IMMUTABLE - no hidden mutations")
    print("✓ Works with torch.compile (state is traced)")
    print("✓ Works with DDP/FSDP (synchronize state explicitly)")
    print("✓ Easy to save/restore training state")
    print("✓ Pure functional - no side effects")
    print("=" * 70)


if __name__ == "__main__":
    main()
