"""Simple DP-SGD training example with the new functional API.

This example demonstrates:
1. Natural composition of clipped_grad() and gaussian_noise()
2. Using .clip_norm attribute for noise calibration
3. Full DP-SGD training loop with the simplified API
4. Swapping in bounded_gaussian_noise() for bounded-domain noise

The new API makes it easy to swap components for research:
- Swap clipping: per_layer_clipped_grad(), adaptive_clipper()
- Swap noise: bounded_gaussian_noise(), correlated_gaussian(), laplace()
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from opaque import bounded_gaussian_noise, clipped_grad, gaussian_noise


def main():
    """Train a simple linear model with DP-SGD."""
    print("=" * 60)
    print("DP-SGD Training with Opaque Functional API")
    print("=" * 60)

    # Hyperparameters
    input_dim = 20
    n_samples = 1000
    batch_size = 32
    epochs = 5
    lr = 0.01
    l2_clip_norm = 1.0
    noise_multiplier = 1.1

    # Generate synthetic data
    print(f"\n1. Generating synthetic data ({n_samples} samples)...")
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
        """MSE loss for linear regression."""
        predictions = x @ params
        return F.mse_loss(predictions, y, reduction="sum")

    # Configure DP-SGD components
    print("\n2. Configuring DP-SGD components...")
    print(f"   - L2 clip norm: {l2_clip_norm}")
    print(f"   - Noise multiplier: {noise_multiplier}")

    # Step 1: Configure gradient clipping
    grad_fn = clipped_grad(
        loss_fn,
        l2_clip_norm=l2_clip_norm,
        batch_argnums=1,  # x and y have batch dimension
    )
    print(f"   - Clip norm from grad_fn: {grad_fn.clip_norm}")

    # Step 2: Configure noise based on clip_norm
    noise_fn, noise_state = gaussian_noise(stddev=noise_multiplier * grad_fn.clip_norm)
    print(f"   - Noise stddev: {noise_multiplier * grad_fn.clip_norm:.3f}")

    # Training loop
    print(f"\n3. Training for {epochs} epochs...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in dataloader:
            # Compute clipped gradients
            grads = grad_fn(params, batch_x, batch_y)

            # Add noise (natural composition!)
            noisy_grads, noise_state = noise_fn(grads, noise_state)

            # Update parameters (simple SGD)
            params = params - lr * noisy_grads

            # Track loss (non-private, for monitoring only)
            with torch.no_grad():
                batch_loss = loss_fn(params, batch_x, batch_y)
                epoch_loss += batch_loss.item()
                n_batches += 1

        avg_loss = epoch_loss / n_batches
        print(f"   Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_loss:.4f}")

    # Evaluate
    print("\n4. Final evaluation...")
    with torch.no_grad():
        predictions = X @ params
        final_loss = F.mse_loss(predictions, y).item()
        print(f"   Final MSE: {final_loss:.4f}")

    print("\n5. Privacy accounting...")
    print("   Note: Use jbr-fed-accounting for actual privacy analysis")
    print(f"   - Noise multiplier: {noise_multiplier}")
    print(f"   - Clip norm: {l2_clip_norm}")
    print(f"   - Steps: {epochs * len(dataloader)}")
    print(f"   - Sample rate: {batch_size / n_samples:.4f}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


def demo_research_flexibility():
    """Demonstrate how easy it is to swap components."""
    print("\n" + "=" * 60)
    print("Research Flexibility Demo")
    print("=" * 60)

    def loss_fn(params, x, y):
        return ((x @ params - y) ** 2).sum()

    # Example 1: Standard DP-SGD
    print("\n1. Standard DP-SGD:")
    grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0)
    _noise_fn, _state = gaussian_noise(stddev=1.1 * grad_fn.clip_norm)
    print(f"   ✓ Clip norm: {grad_fn.clip_norm}")
    print(f"   ✓ Noise: Gaussian(stddev={1.1 * grad_fn.clip_norm})")

    # Example 2: Different clip norm
    print("\n2. Higher clip norm:")
    grad_fn_2 = clipped_grad(loss_fn, l2_clip_norm=2.0)
    _noise_fn, _state = gaussian_noise(stddev=1.1 * grad_fn_2.clip_norm)
    print(f"   ✓ Clip norm: {grad_fn_2.clip_norm}")
    print(f"   ✓ Noise: Gaussian(stddev={1.1 * grad_fn_2.clip_norm})")

    # Example 3: Rescale to unit norm
    print("\n3. Unit norm (rescale_to_unit_norm=True):")
    grad_fn_3 = clipped_grad(loss_fn, l2_clip_norm=5.0, rescale_to_unit_norm=True)
    _noise_fn, _state = gaussian_noise(stddev=1.1 * grad_fn_3.clip_norm)
    print(f"   ✓ Clip norm: {grad_fn_3.clip_norm}")  # Should be 1.0
    print(f"   ✓ Noise: Gaussian(stddev={1.1 * grad_fn_3.clip_norm})")

    # Example 4: Bounded Gaussian noise (truncated normal)
    print("\n4. Bounded Gaussian (Chen & Hale, 2024):")
    _noise_fn, _state = bounded_gaussian_noise(
        stddev=1.1 * grad_fn.clip_norm, bounds=(-3.0, 3.0)
    )
    print(
        f"   ✓ Noise: BoundedGaussian(stddev={1.1 * grad_fn.clip_norm}, bounds=(-3, 3))"
    )
    print("   → Outputs guaranteed in [-3.0, 3.0]")

    # Example 5: Future - swap clipping mechanism
    print("\n5. Future: Swappable mechanisms:")
    print("   # noise_fn = correlated_gaussian_noise(stddev=1.1, rank=10)")
    print("   # grad_fn = per_layer_clipped_grad(loss_fn, clip_norms={...})")
    print("   # grad_fn = adaptive_clipper(loss_fn, target_quantile=0.5)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
    demo_research_flexibility()
