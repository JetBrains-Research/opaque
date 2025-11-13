"""Integration tests for optimizer equivalence.

Tests that DP optimizers produce correct gradients and updates by comparing:
1. Standard PyTorch optimizer (torch.optim.AdamW) with standard gradients
2. Standard PyTorch optimizer with clipped_grad (no noise)
3. Opaque DP optimizer with clipped_grad (no noise)

This validates that gradient clipping and DP optimizer logic work correctly
when noise_multiplier=0 (no privacy, just correctness testing).
"""

import pytest
import torch
from torch.optim import AdamW

from opaque.clipping import clipped_grad
from opaque.optimizers import dp_adam_ac
from opaque.utils import make_functional
# Import shared fixtures
from tests.integration.conftest import (
    compute_causal_lm_loss,
    create_custom_llama,
    create_huggingface_model,
)

# Optional: Import transformers if available
try:
    import transformers

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# ============================================================================
# Test Parametrization
# ============================================================================

# Test against same models as gradient_equivalence tests
TEST_MODELS = [
    pytest.param("custom_llama", id="custom_llama"),
    pytest.param(
        "gpt2",
        marks=[
            pytest.mark.skipif(not HAS_TRANSFORMERS, reason="requires transformers"),
        ],
        id="gpt2",
    ),
    pytest.param(
        "Qwen/Qwen2.5-0.5B",
        marks=[
            pytest.mark.skipif(not HAS_TRANSFORMERS, reason="requires transformers"),
            pytest.mark.slow()
        ],
        id="qwen2.5-0.5b",
    ),
    pytest.param(
        "google/gemma-3-270m",
        marks=[
            pytest.mark.skipif(not HAS_TRANSFORMERS, reason="requires transformers"),
            pytest.mark.slow()
        ],
        id="gemma-3",
    ),
]


# ============================================================================
# Gradient Computation Methods
# ============================================================================


def compute_grads_standard(model, x_batch, y_batch):
    """Method 1: Standard PyTorch backward pass.

    Returns:
        grads: Dictionary of gradients {param_name: grad_tensor}
        loss: Scalar loss value
    """
    model.zero_grad()
    predictions = model(x_batch)
    loss = compute_causal_lm_loss(predictions, y_batch)
    loss.backward()

    grads = {
        name: param.grad.clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }

    return grads, loss.item()


def compute_grads_clipped(model, x_batch, y_batch, clip_norm=1.0):
    """Method 2: Functional API with clipped_grad (no noise).

    Returns:
        grads: Dictionary of averaged gradients {param_name: grad_tensor}
        loss: Average loss across batch
    """
    fmodel, params = make_functional(model, disable_autograd_tracking=True)
    param_names = [name for name, _ in model.named_parameters()]

    def per_example_loss_fn(params_tuple, x_single, y_single):
        # Add batch dimension
        x_batch_single = x_single.unsqueeze(0)
        y_batch_single = y_single.unsqueeze(0)

        predictions = fmodel(params_tuple, x_batch_single)
        return compute_causal_lm_loss(predictions, y_batch_single)

    # Use clipped_grad with large clip norm (effectively no clipping for comparison)
    clipped_grad_fn = clipped_grad(
        per_example_loss_fn,
        argnums=0,
        batch_argnums=(1, 2),
        l2_clip_norm=clip_norm,
        keep_batch_dim=False,
        return_grad_norms=True,
        return_values=True,
    )

    grads_tuple, aux = clipped_grad_fn(params, x_batch, y_batch)

    # Average gradients
    batch_size = len(x_batch)
    grads_tuple_avg = tuple(grad / batch_size for grad in grads_tuple)
    grads = {name: grad for name, grad in zip(param_names, grads_tuple_avg)}

    loss_avg = aux.values.mean().item()

    return grads, loss_avg


def compute_step_pytorch_adamw(model, grads, lr=0.001, weight_decay=0.01):
    """Apply AdamW update using PyTorch optimizer.

    Args:
        model: PyTorch model
        grads: Dictionary of gradients
        lr: Learning rate
        weight_decay: Weight decay

    Returns:
        new_params: Dictionary of updated parameters
    """
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Set gradients
    for name, param in model.named_parameters():
        if name in grads:
            param.grad = grads[name]

    # Take optimizer step
    optimizer.step()

    # Return new parameters
    new_params = {name: param.data.clone() for name, param in model.named_parameters()}

    return new_params


def compute_step_dp_adamac(model, x_batch, y_batch, clip_norm=1.0, lr=0.001, noise_multiplier=0.0, seed=42):
    """Apply DP-Adam-AC update using Opaque optimizer.

    Note: DP-Adam-AC doesn't have weight_decay (uses DP noise as implicit regularization).

    Args:
        model: PyTorch model
        x_batch: Input batch
        y_batch: Target batch
        clip_norm: Gradient clipping norm
        lr: Learning rate
        noise_multiplier: Noise multiplier (0 for no noise)
        seed: Random seed for noise generation

    Returns:
        new_params: Dictionary of updated parameters
        state: Optimizer state
    """
    fmodel, params = make_functional(model, disable_autograd_tracking=True)
    param_names = [name for name, _ in model.named_parameters()]

    def per_example_loss_fn(params_tuple, x_single, y_single):
        x_batch_single = x_single.unsqueeze(0)
        y_batch_single = y_single.unsqueeze(0)
        predictions = fmodel(params_tuple, x_batch_single)
        return compute_causal_lm_loss(predictions, y_batch_single)

    # Compute clipped gradients
    clipped_grad_fn = clipped_grad(
        per_example_loss_fn,
        argnums=0,
        batch_argnums=(1, 2),
        l2_clip_norm=clip_norm,
        keep_batch_dim=False,
        return_grad_norms=True,
        return_values=True,
    )

    grads_tuple, aux = clipped_grad_fn(params, x_batch, y_batch)
    grad_norms = aux.grad_norms

    # Create DP-Adam-AC optimizer
    # sample_rate = batch_size / dataset_size (we'll use a dummy value for testing)
    batch_size = len(x_batch)
    dataset_size = 1000  # Dummy value
    sample_rate = batch_size / dataset_size

    init_fn, step_fn = dp_adam_ac(
        learning_rate=lr,
        initial_clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
        sample_rate=sample_rate,
        target_delta=1e-5,
        seed=seed,
    )

    # Initialize optimizer state
    state = init_fn(params)

    # Take optimizer step
    batch_sizes = torch.ones(len(x_batch))
    new_params, new_state, metrics = step_fn(params, grads_tuple, grad_norms, batch_sizes, state)

    # Convert back to dictionary
    new_params_dict = {name: param for name, param in zip(param_names, new_params)}

    return new_params_dict, new_state


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.parametrize("model_id", TEST_MODELS)
class TestOptimizerEquivalence:
    """Test optimizer equivalence across different methods."""

    def test_gradients_match_with_large_clip_norm(self, model_id):
        """Verify clipped_grad matches standard gradients with large clip norm."""
        # Setup
        torch.manual_seed(42)

        # Create model and data based on model_id
        if model_id == "custom_llama":
            model, x_batch = create_custom_llama()
            y_batch = x_batch  # Causal LM: predict next token
        else:
            model, x_batch = create_huggingface_model(model_id)
            y_batch = x_batch  # Causal LM: predict next token

        # Very large clip norm (no actual clipping)
        large_clip_norm = 1e6

        # Method 1: Standard gradients
        grads_standard, loss_standard = compute_grads_standard(model, x_batch, y_batch)

        # Method 2: Clipped gradients (same model, will recompute gradients)
        grads_clipped, loss_clipped = compute_grads_clipped(
            model, x_batch, y_batch, clip_norm=large_clip_norm
        )

        # Verify losses match
        assert abs(loss_standard - loss_clipped) < 1e-5, (
            f"Losses don't match: {loss_standard} vs {loss_clipped}"
        )

        # Verify gradients match (use looser tolerance for HF models)
        atol, rtol = (5e-4, 5e-4) if model_id != "custom_llama" else (5e-5, 5e-5)
        for name in grads_standard.keys():
            assert torch.allclose(
                grads_standard[name], grads_clipped[name], atol=atol, rtol=rtol
            ), f"Gradients don't match for {name}"

        print("\n✓ Gradients match between standard and clipped_grad (large clip norm)")

    def test_optimizer_updates_match_without_clipping(self, model_id):
        """Verify DP-Adam-AC matches PyTorch AdamW when clip_norm is large and noise=0.

        Note: We compare without weight_decay since DP-Adam-AC doesn't have it
        (DP noise acts as implicit regularization).
        """
        # Setup
        torch.manual_seed(42)
        lr = 0.001
        large_clip_norm = 1e6

        # Method 1: Standard PyTorch AdamW (no weight decay for fair comparison)
        torch.manual_seed(42)  # Reset for consistency
        if model_id == "custom_llama":
            model_standard, x_batch = create_custom_llama()
            y_batch = x_batch  # Causal LM: predict next token
        else:
            model_standard, x_batch = create_huggingface_model(model_id)
            y_batch = x_batch  # Causal LM: predict next token

        grads_standard, _ = compute_grads_standard(model_standard, x_batch, y_batch)
        params_standard = compute_step_pytorch_adamw(
            model_standard, grads_standard, lr=lr, weight_decay=0.0
        )

        # Method 2: DP-Adam-AC with no noise and large clip norm
        torch.manual_seed(42)  # Reset for same initialization
        if model_id == "custom_llama":
            model_dp, x_batch_dp = create_custom_llama()
            y_batch_dp = x_batch_dp  # Causal LM: predict next token
        else:
            model_dp, x_batch_dp = create_huggingface_model(model_id)
            y_batch_dp = x_batch_dp  # Causal LM: predict next token

        params_dp, _ = compute_step_dp_adamac(
            model_dp, x_batch_dp, y_batch_dp,
            clip_norm=large_clip_norm,
            lr=lr,
            noise_multiplier=0.0,
        )

        # Compare parameters (should be similar but not identical due to implementation differences)
        print("\n" + "=" * 80)
        print("Parameter comparison: PyTorch AdamW vs DP-Adam-AC (no noise, no clipping)")
        print("=" * 80)

        for name in params_standard.keys():
            diff = (params_standard[name] - params_dp[name]).abs().max().item()
            rel_diff = diff / params_standard[name].abs().max().item() if params_standard[
                                                                              name].abs().max().item() > 0 else 0
            print(f"{name:20s}: max_abs_diff={diff:.6e}, rel_diff={rel_diff:.6e}")

            # Use more lenient tolerance since optimizer implementations may differ slightly
            # Larger models need even more tolerance due to accumulated numerical differences
            atol = 5e-3 if model_id != "custom_llama" else 1e-3
            rtol = 5e-3 if model_id != "custom_llama" else 1e-3
            assert torch.allclose(
                params_standard[name], params_dp[name], atol=atol, rtol=rtol
            ), f"Parameters don't match for {name}: diff={diff:.6e}"

        print("✓ Optimizer updates match within tolerance")

    def test_clipping_affects_gradients(self, model_id):
        """Verify that gradient clipping actually clips per-example gradients.

        Note: We're testing that clipped_grad clips *per-example* gradients,
        so the summed/averaged gradient norm will be different from clip_norm.
        """
        # Setup
        torch.manual_seed(42)

        # Create model and data based on model_id
        if model_id == "custom_llama":
            model, x_batch = create_custom_llama()
            y_batch = x_batch  # Causal LM: predict next token
        else:
            model, x_batch = create_huggingface_model(model_id)
            y_batch = x_batch  # Causal LM: predict next token

        # Small clip norm to trigger clipping
        small_clip_norm = 0.5

        # Method 1: No clipping
        grads_no_clip, _ = compute_grads_clipped(model, x_batch, y_batch, clip_norm=1e6)

        # Method 2: With clipping
        grads_clipped, _ = compute_grads_clipped(model, x_batch, y_batch, clip_norm=small_clip_norm)

        # Compute norms
        def dict_norm(grads_dict):
            return torch.sqrt(sum((g ** 2).sum() for g in grads_dict.values()))

        norm_no_clip = dict_norm(grads_no_clip).item()
        norm_clipped = dict_norm(grads_clipped).item()

        print("\n" + "=" * 80)
        print("Clipping verification")
        print("=" * 80)
        print(f"Avg gradient norm without clipping: {norm_no_clip:.6f}")
        print(f"Avg gradient norm with clipping:    {norm_clipped:.6f}")
        print(f"Per-example clip norm threshold:    {small_clip_norm:.6f}")
        print(f"Reduction ratio:                     {norm_clipped / norm_no_clip:.6f}")

        # Verify clipping occurred (summed gradient norm should be smaller)
        assert norm_clipped < norm_no_clip, (
            f"Clipping didn't reduce gradient norm: {norm_clipped} >= {norm_no_clip}"
        )
        # Per-example clipping should reduce the norm, but not to exactly clip_norm
        # because we're averaging multiple clipped examples
        assert norm_clipped < norm_no_clip * 0.9, "Clipping had minimal effect"

        print("✓ Gradient clipping reduces gradient norms correctly")

    def test_dp_adamac_with_noise_adds_randomness(self, model_id):
        """Verify that noise_multiplier > 0 adds noise to gradients."""
        # Setup
        torch.manual_seed(42)

        # Create model and data based on model_id
        if model_id == "custom_llama":
            model, x_batch = create_custom_llama()
            y_batch = x_batch  # Causal LM: predict next token
        else:
            model, x_batch = create_huggingface_model(model_id)
            y_batch = x_batch  # Causal LM: predict next token

        clip_norm = 1.0
        noise_multiplier = 1.0

        # Run twice with same inputs but different seeds
        params_run1, _ = compute_step_dp_adamac(
            model, x_batch, y_batch,
            clip_norm=clip_norm,
            noise_multiplier=noise_multiplier,
            seed=42,
        )

        # Reset model and run again with different seed
        if model_id == "custom_llama":
            model2, _ = create_custom_llama()
        else:
            model2, _ = create_huggingface_model(model_id)

        params_run2, _ = compute_step_dp_adamac(
            model2, x_batch, y_batch,
            clip_norm=clip_norm,
            noise_multiplier=noise_multiplier,
            seed=123,  # Different seed
        )

        # Verify parameters differ (noise adds randomness)
        print("\n" + "=" * 80)
        print("Noise verification")
        print("=" * 80)

        max_diff = 0.0
        for name in params_run1.keys():
            diff = (params_run1[name] - params_run2[name]).abs().max().item()
            max_diff = max(max_diff, diff)
            print(f"{name:20s}: max_abs_diff={diff:.6e}")

        assert max_diff > 1e-6, "No difference detected - noise may not be working"
        print(f"✓ Noise adds randomness (max diff: {max_diff:.6e})")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
